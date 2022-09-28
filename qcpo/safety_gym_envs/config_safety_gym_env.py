
config0 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'button',
    'observe_goal_lidar': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,

    'constrain_hazards': True,
    'constrain_buttons': True,
    'observe_hazards': True,
    'observe_buttons': True,

    'buttons_num': 2,
    'buttons_size': 0.1,
    'buttons_keepout': 0.2,
    'buttons_locations': [(-1, -1), (1, 1)],
    'hazards_num': 3,
    'hazards_size': 0.3,
    'hazards_keepout': 0.305,
    'hazards_locations': [(0, 0), (-1, 1), (0.5, -0.5)],
}

config1 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 3,
    'hazards_size': 0.3,
    'hazards_keepout': 0.305,
}

config2 = {
    'placements_extents': [-2, -2, 2, 2],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'constrain_gremlins': True,
    'observe_gremlins': True,
    'lidar_max_dist': 5,
    'lidar_num_bins': 16,
    'hazards_num': 5,
    'hazards_size': 0.3,
    'hazards_keepout': 0.305,
    'gremlins_num': 3,
    'gremlins_travel': 0.35,
    'gremlins_keepout': 0.4,
}

config3 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'button',
    'observe_goal_lidar': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,

    'constrain_buttons': True,
    'observe_buttons': True,

    'buttons_num': 6,
    'buttons_size': 0.1,
    'buttons_keepout': 0.2,

}

