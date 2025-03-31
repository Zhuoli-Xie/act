import os
import sys
import torch
import pickle
import argparse
import time
import threading
from policy import ACTPolicy, CNNMLPPolicy
import numpy as np
from utils import set_seed
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import deque
from einops import rearrange
import getch

# autopep8: off
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')             
sys.path.append(libs_path)  
from RobotLib import Robot
# autopep8: on

# robot = Robot("192.168.11.3:50051", "", "")
# robot.set_arm_enable(1, 1)
# robot.set_arm_mode(1, 1) 
# robot.set_arm_state(1, 0)
# robot.set_arm_enable(2, 1)
# robot.set_arm_mode(2, 1)
# robot.set_arm_state(2, 0)

# from xarm.wrapper import XArmAPI
# left_arm_xarm = XArmAPI('192.168.11.11')
# left_arm_xarm.clean_warn()
# left_arm_xarm.clean_error()
# left_arm_xarm.motion_enable(enable=True)
# left_arm_xarm.set_mode(1)
# time.sleep(0.3)
# left_arm_xarm.set_state(state=0)
# time.sleep(0.01)


USE_REAL_CAMERA = True

left_images_deque = deque(maxlen=5)
right_images_deque = deque(maxlen=5)
left_lock = threading.Lock()
right_lock = threading.Lock()
joint_states_deque = deque(maxlen=5)
gripper_deque = deque(maxlen=5)
js_lock = threading.Lock()
gripper_lock = threading.Lock()

def get_char():
    return getch.getch()
    #return

def evaluate(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    from constants import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 15
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 6
        dec_layers = 9
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
    }

    ckpt_names = [f'policy_best.ckpt']
    for ckpt_name in ckpt_names:
        eval_bc(config, ckpt_name, save_episode=True)
    print()
    exit()

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def get_image(camera_names):
    curr_images = []
    for cam_name in camera_names:
        if cam_name == 'left':
            with left_lock:
                curr_image = rearrange(left_images_deque[-1], 'h w c -> c h w')
            curr_images.append(curr_image)
        elif cam_name == 'right':
            with right_lock:
                curr_image = rearrange(right_images_deque[-1], 'h w c -> c h w')
            curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    id = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    max_timesteps = int(max_timesteps * 3) # may increase for real-world tasks

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

    qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
    image_list = [] # for visualization
    qpos_list = []
    target_qpos_list = []

    with torch.inference_mode():
        for t in range(max_timesteps):
            # success, js_position, velocity, effort  = robot.get_joint_state(id)
            # success, g_position, torque = robot.get_gripper_position(1)
            with js_lock:
                js_position = joint_states_deque[-1]
            with gripper_lock:
                g_position = gripper_deque[-1]
            obs_js = np.append(js_position, g_position.data)  
            qpos_numpy = np.array(obs_js)
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_history[:, t] = qpos
            curr_image = get_image(camera_names)

            ### query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                    print(all_actions.shape)
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
            else:
                raise NotImplementedError

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            target_qpos = action

            ### send joints
            print(f"target_qpos: {target_qpos}")
            l_angles = target_qpos[:7]
            r_angles = target_qpos[7:13]
            g_angles = target_qpos[14]
            speed = 0
            acc = 0
            wait = 0

            # 这儿需要插值吗
            # robot.set_arm_servo_angle_j(1, l_angles, speed, acc, wait)
            # robot.set_arm_servo_angle_j(2, r_angles, speed, acc, wait)
            # robot.set_gripper_position(1, g_angles)

            ### for visualization
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            print(f"============{t=}==============")

            if t % 30 == 0:
                print(f".....................")

            # 要保存相机视频吗
            # if save_episode:
            #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

class CameraNode(Node):
    def __init__(self, name, is_debug=False):
        super().__init__(name)
        from std_msgs.msg import Float32
        from sensor_msgs.msg import JointState
        js_callback_func = self.js_func
        g_callback_func = self.g_func
        self.create_subscription(JointState, f"/joint_states", js_callback_func, 10)
        self.create_subscription(Float32, f"/gripper_state/left", g_callback_func, 10)
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ['right', 'left']
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None) 
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'right':
                callback_func = self.image_cb_cam_right
            elif cam_name == 'left':
                callback_func = self.image_cb_cam_left
            else:
                raise NotImplementedError
            self.create_subscription(Image, f"/{cam_name}/color/image_raw", callback_func, 10) # edit it
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.sec)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nanosec)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.sec + data.header.stamp.nanosec * 1e-9)

    def image_cb_cam_right(self, data):
        cam_name = 'right'
        self.image_cb(cam_name, data)
        with left_lock:
            left_images_deque.append(getattr(self, f'{cam_name}_image'))

    def image_cb_cam_left(self, data):
        cam_name = 'left'
        self.image_cb(cam_name, data)
        with right_lock:
            right_images_deque.append(getattr(self, f'{cam_name}_image'))

    def js_func(self, data):
        with js_lock:
            joint_states_deque.append(data.position[5:19])

    def g_func(self, data):
        with gripper_lock:
            gripper_deque.append(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    if USE_REAL_CAMERA:
        rclpy.init()
        node = CameraNode("camera_node")
        thread_node = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        thread_node.start()
    
    try:
        while True:
            with left_lock:
                left_len = len(left_images_deque)
            with right_lock:
                right_len = len(right_images_deque)
            with js_lock:
                js_len = len(joint_states_deque)
            with gripper_lock:
                g_len = len(gripper_deque)
            if left_len == 5 and right_len == 5 and js_len == 5 and g_len == 5:
                print(f"当前队列长度：{(left_len, right_len, js_len, g_len)}") 
                break
            else:
                print(f"当前队列长度：{(left_len, right_len, js_len, g_len)}") 

            time.sleep(0.5) 
                
        print("队列已满，进入下一步")
        evaluate(vars(parser.parse_args()))
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()