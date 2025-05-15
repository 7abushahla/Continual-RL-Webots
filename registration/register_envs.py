from gym.envs.registration import register
from envs.robot_controller import EpuckRobot

register(
    id='EpuckObstacle-v0',
    entry_point='envs.supervisor_envs:EpuckObstacleSupervisorEnv',
    kwargs={'robot_controller': EpuckRobot}
)

register(
    id='EpuckLine-v0',
    entry_point='envs.supervisor_envs:EpuckLineFollowSupervisorEnv',
    kwargs={'robot_controller': EpuckRobot}
)

register(
    id='EpuckMaze-v0',
    entry_point='envs.supervisor_envs:EpuckMazeSupervisorEnv',
    kwargs={'robot_controller': EpuckRobot}
)
