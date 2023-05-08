import numpy as np
from collections import deque
from team_code.render import render, render_self_car, find_peak_box
import math

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

def downsample_waypoints(waypoints, precision=0.2):
    """
    waypoints: [float lits], 10 * 2, m
    """
    downsampled_waypoints = []
    downsampled_waypoints.append(np.array([0, 0]))
    last_waypoint = np.array([0.0, 0.0])
    for i in range(10):
        now_waypoint = waypoints[i]
        dis = np.linalg.norm(now_waypoint - last_waypoint)
        if dis > precision:
            interval = int(dis / precision)
            move_vector = (now_waypoint - last_waypoint) / (interval + 1)
            for j in range(interval):
                downsampled_waypoints.append(last_waypoint + move_vector * (j + 1))
        downsampled_waypoints.append(now_waypoint)
        last_waypoint = now_waypoint
    return downsampled_waypoints

def collision_detections(map1, map2, threshold=0.04):
    """
    map1: rendered surround vehicles
    map2: self-car
    """
    assert map1.shape == map2.shape
    overlap_map = (map1 > 0.01) & (map2 > 0.01)
    ratio = float(np.sum(overlap_map)) / np.sum(map2 > 0)
    ratio2 = float(np.sum(overlap_map)) / np.sum(map1 > 0)
    if ratio < threshold:
        return True
    else:
        return False

def get_max_safe_distance(meta_data, downsampled_waypoints, t, collision_buffer, threshold):
    surround_map = render(meta_data.reshape(20, 20, 7), t=t)[0][:100, 40:140]
    if np.sum(surround_map) < 1:
        return np.linalg.norm(downsampled_waypoints[-3])
    # need to render self-car map
    hero_bounding_box = np.array([2.45, 1.0]) + collision_buffer
    safe_distance = 0.0
    for i in range(len(downsampled_waypoints) - 2):
        aim = (downsampled_waypoints[i + 1] + downsampled_waypoints[i + 2]) / 2.0
        loc = downsampled_waypoints[i]
        ori = aim - loc
        self_car_map = render_self_car(loc=loc, ori=ori, box=hero_bounding_box)[
            :100, 40:140
        ]
        if collision_detections(surround_map, self_car_map, threshold) is False:
            break
        safe_distance = max(safe_distance, np.linalg.norm(loc))
    return safe_distance

class InterfuserController(object):
    def __init__(self, config):
        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP,
            K_I=config.speed_KI,
            K_D=config.speed_KD,
            n=config.speed_n,
        )
        self.collision_buffer = np.array(config.collision_buffer)
        self.config = config
        self.detect_threshold = config.detect_threshold
        self.stop_steps = 0
        self.forced_forward_steps = 0

        self.red_light_steps = 0
        self.block_red_light = 0

        self.in_stop_sign_effect = False
        self.block_stop_sign_distance = (
            0  # If this is 3 here, it means in 3m, stop sign will not take effect again
        )
        self.stop_sign_trigger_times = 0

        self.prev_timestamp = 0

    def run_step(
        self, speed, waypoints, junction, traffic_light_state, stop_sign, meta_data, timestamp
    ):
        """
        speed: int, m/s
        waypoints: [float lits], 10 * 2, m
        junction: float, prob of the vehicle not at junction
        traffic_light_state: float, prob of the traffic light state is Red or Yellow
        stop_sign: float, prob of not at stop_sign
        meta_data: 20 * 20 * 7
        """
        if speed < 0.2:
            self.stop_steps += 1
        else:
            self.stop_steps = max(0, self.stop_steps - 10)

        if speed < 0.06 and self.in_stop_sign_effect:
            self.in_stop_sign_effect = False

        if junction < 0.3:
            self.stop_sign_trigger_times = 0

        if traffic_light_state > 0.7:
            self.red_light_steps += 1
        else:
            self.red_light_steps = 0
        if self.red_light_steps > 1000:
            self.block_red_light = 80
            self.red_light_steps = 0
        if self.block_red_light > 0:
            self.block_red_light -= 1
            traffic_light_state = 0.01

        if stop_sign < 0.6 and self.block_stop_sign_distance < 0.1:
            self.in_stop_sign_effect = True
            self.block_stop_sign_distance = 2.0
            self.stop_sign_trigger_times = 3

        self.block_stop_sign_distance = max(
            0, self.block_stop_sign_distance - 0.05 * speed
        )
        if self.block_stop_sign_distance < 0.1:
            if self.stop_sign_trigger_times > 0:
                self.block_stop_sign_distance = 2.0
                self.stop_sign_trigger_times -= 1
                self.in_stop_sign_effect = True


        aim = (waypoints[1] + waypoints[0]) / 2.0
        aim[1] *= -1
        heading_error = np.pi / 2 - np.arctan2(aim[1], aim[0])
        if speed < 0.01:
            heading_error = 0




        '''
        ################################################################# Bang Bang Controller ###############################################################################
        x = 0
        y = 0
        P = np.asarray([x, y])
        for i in range(len(waypoints)):
            dis = self.get_distance(x, y, waypoints[i][0], waypoints[i][1])
            if abs(dis - lookahead_dis) <= self._eps_lookahead:
                return i
        return len(waypoints)-1
        i = self.get_lookahead_point_index(x, y, waypoints, self._cte_ref_dist) # Get current waypoint index
        if i == 0:
            A = np.asarray([waypoints[i][0], waypoints[i][1]])
            B = np.asarray([waypoints[i+1][0], waypoints[i+1][1]])
        else:
            A = np.asarray([waypoints[i-1][0], waypoints[i-1][1]])
            B = np.asarray([waypoints[i][0], waypoints[i][1]])
        n = B-A
        m = P-A
        dirxn = self.get_steering_direction(n, m)
        crosstrack_error = dirxn*(np.abs(((B[0]-A[0])*(A[1]-P[1]))-((A[0]-P[0])*(B[1]-A[1])))/np.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2))
        crosstrack_error = self.get_crosstrack_error(x, y, waypoints)
        if crosstrack_error > 0:
            steering = 1.22*0.1
        elif crosstrack_error < 0:
            steering = -1.22*0.1
        else:
            steering = 0
        ###############################################################################################################################################################################
        '''
        ############################################################## MPC #########################################################################################################
        # MPC control
        # Discrete steering angle from -1.2 to 1.2 with interval of 0.1.
        steer_list=np.arange(-1.2,1.2,0.1)
        j_min = 0
        steer_output = 0
        for idx in range(len(steer_list)):
            vehicle_heading_yaw = np.pi/2 + steer_list[idx]
            t_diff = timestamp - self.prev_timestamp
            pred_x = 0 + speed*t_diff*np.cos(vehicle_heading_yaw)
            pred_y = 0 + speed*t_diff*np.sin(vehicle_heading_yaw)
            delta_dis = math.sqrt((waypoints[-1][0] - pred_x)**2 + (waypoints[-1][1] - pred_y)**2)                 
            j = 0.1*delta_dis**2 + steer_list[idx]**2
            if idx == 0:
                j_min = j
            if j < j_min:
                j_min = j
                steer_output = steer_list[idx]
        # Obey the max steering angle bounds
        if steer_output > 1.22:
            steer_output = 1.22
        if steer_output < -1.22:
            steer_output = -1.22
            





        ###############################################################################################################################################################################

        ########################################################## STANLEY CONTROLLER #################################################################################################
        '''
        # crosstrack error
        e_r = 0
        min_idx = 0
        # Get the minmum distance between the vehicle and target trajectory
        for idx in range(len(waypoints)):
            dis = np.linalg.norm(waypoints[idx])
            if idx == 0:
                e_r = dis
            if dis < e_r:
                e_r = dis
                min_idx = idx
        min_path_yaw = np.arctan(waypoints[min_idx][1]/waypoints[min_idx][0])
        cross_yaw_error = min_path_yaw - (np.pi/2)
        if cross_yaw_error > np.pi/2:
            cross_yaw_error -= np.pi
        if cross_yaw_error < - np.pi/2:
            cross_yaw_error += np.pi 
        if cross_yaw_error > 0:
            e_r = e_r
        else:
            e_r = -e_r
        delta_error = np.arctan(0.1*e_r/(speed+1.0e-6))        
        steer_output = heading_error + delta_error
        print("steer: "+str(steer_output))
        if steer_output>1.22:
            steer_output=1.22
        if steer_output<-1.22:
            steer_output=-1.22
        '''
        #################################################################################################################################################################################



        ########################################################## PURE PERSUIT CONTROL #################################################################################################
        '''
        # Pure Persuit Control
        y_delta=aim[1]
        x_delta=aim[0]
        alpha=np.arctan(y_delta/x_delta)-(np.pi/2)
        if alpha > np.pi/2:
            alpha -= np.pi
        if alpha < - np.pi/2:
            alpha += np.pi 
        
        steer_output=np.arctan(2*np.sin(alpha)/(15*speed))
        # Obey the max steering angle bounds
        if steer_output>1.22:
            steer_output=1.22
        if steer_output<-1.22:
            steer_output=-1.22
        '''
        ####################################################################################################################################################################################


        #print("1:",steer_output)



        
        steer = -np.degrees(steer_output*2) / 90

        #print("2:",steer)
        #steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        #print("3:",steer)

        brake = False
        # get desired speed
        downsampled_waypoints = downsample_waypoints(waypoints)
        d_0 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=0,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_05 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=0.5,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_075 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=0.75,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_1 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=1,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_15 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=1.5,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_2 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=2,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )

        d_05 = min(d_0, d_05, d_075)
        d_1 = min(d_05, d_075, d_15, d_2)

        safe_dis = min(d_05, d_1)
        d_0 = max(0, d_0 - 2.0)
        d_05 = max(0, d_05 - 2.0)
        d_1 = max(0, d_1 - 2.0)

        if d_0 < max(3, speed):
            brake = True
            desired_speed = 0.0
        else:
            desired_speed = max(
                0,
                min(
                    4 * d_05 - speed - max(0, speed - 2.5),
                    self.config.max_speed,
                    2 * d_1 - 0.5 * speed - max(0, speed - 2.5),
                ),
            )
            if junction > 0.0 and traffic_light_state > 0.3:
                brake = True
                desired_speed = 0.0
        desired_speed = desired_speed if brake is False else 0.0

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)

        if speed > desired_speed * self.config.brake_ratio:
            brake = True

        '''
        meta_info_1 = "d0:%.1f, d05:%.1f, d1:%.1f, desired_speed:%.2f" % (
            d_0,
            d_05,
            d_1,
            desired_speed,
        )
        '''
        meta_info_1 = "speed: %.2f, target_speed: %.2f" % (
            speed,
            desired_speed,
        )
        meta_info_2 = "on_road_prob: %.2f, red_light_prob: %.2f, stop_sign_prob: %.2f" % (
            junction,
            traffic_light_state,
            1 - stop_sign,
        )
        meta_info_3 = "stop_steps:%d, block_stop_sign_distance:%.1f" % (
            self.stop_steps,
            self.block_stop_sign_distance,
        )

        if self.stop_steps > 1200:
            self.forced_forward_steps = 12
            self.stop_steps = 0
        if self.forced_forward_steps > 0:
            throttle = 0.8
            brake = False
            self.forced_forward_steps -= 1
        if self.in_stop_sign_effect:
            throttle = 0
            brake = True

        self.prev_timestamp = timestamp

        return steer, throttle, brake, (meta_info_1, meta_info_2, meta_info_3, safe_dis)
