from controller import Supervisor, Emitter
import random
import numpy as np

class Grid:
    def __init__(self, width, height, origin, cell_size):
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.origin = origin
        self.cell_size = cell_size
        
    def size(self):
        return len(self.grid[0]), len(self.grid)
        
    def empty(self):
        self.grid = [[None for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        
    def add_cell(self, x, y, node, z=None):
        if not self.is_in_range(x, y):
            # print(f"[Grid] Warning: Tried to place node out of range at ({x}, {y})")
            return False
        if self.grid[y][x] is not None:
            # print(f"[Grid] Warning: Cell ({x}, {y}) already occupied by {self.grid[y][x].getDef()}")
            return False
        # Prevent placing the same node multiple times
        for row in self.grid:
            if node in row:
                # print(f"[Grid] Warning: Node {node.getDef()} already placed elsewhere in grid")
                return False
        wx, wy = self.get_world_coordinates(x, y)
        if wx is None or wy is None:
            # print(f"[Grid] Warning: Could not get world coordinates for ({x}, {y})")
            return False
        if z is None:
            z = node.getPosition()[2]
        self.grid[y][x] = node
        node.getField("translation").setSFVec3f([wx, wy, z])
        print(f"[Grid] Placing {node.getDef()} at ({x},{y}) -> {wx}, {wy}, {z}")
        return (x, y)  # Return the cell coordinates
        
    def is_in_range(self, x, y):
        return (0 <= x < len(self.grid[0])) and (0 <= y < len(self.grid))
        
    def get_world_coordinates(self, x, y):
        if self.is_in_range(x, y):
            world_x = self.origin[0] + x * self.cell_size[0]
            world_y = self.origin[1] - y * self.cell_size[1]
            return world_x, world_y
        else:
            return None, None
            
    def add_random(self, node, z=None):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] == node:
                    self.grid[y][x] = None
        empty_positions = []
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] is None:
                    empty_positions.append((x, y))
        if not empty_positions:
            # print(f"[Grid] Warning: No suitable empty positions found for {node.getDef()}")
            return False
        random.shuffle(empty_positions)
        for pos in empty_positions:
            cell = self.add_cell(pos[0], pos[1], node, z=z)
            if cell:
                return cell  # Return the cell coordinates
        # print(f"[Grid] Warning: Could not place {node.getDef()} after trying all positions")
        return None
        
    def find_by_name(self, name):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] and self.grid[y][x].getField("name").getSFString() == name:
                    return x, y
        return None

    def bfs_path(self, start, goal):
        """Breadth-First Search for shortest path on the grid, allowing start and goal cells to be traversed even if occupied."""
        from collections import deque
        queue = deque([(start, [start])])
        visited = set([start])
        while queue:
            (current, path) = queue.popleft()
            if current == goal:
                return path
            x, y = current
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if not self.is_in_range(nx, ny):
                    continue
                # Allow traversing start and goal cells even if occupied
                if (nx, ny) == goal or (nx, ny) == start or self.grid[ny][nx] is None:
                    if (nx, ny) not in visited:
                        queue.append(((nx, ny), path+[(nx, ny)]))
                        visited.add((nx, ny))
        return None

class RandomizerSupervisor(Supervisor):
    def __init__(self, number_of_obstacles=10, seed=None):
        super().__init__()
        self.emitter = self.getDevice('emitter')
        # Set emitter channel to 1 for communication with RL supervisor
        self.emitter.setChannel(1)
        # print(f"[RandomizerSupervisor] Emitter initialized on channel {self.emitter.getChannel()}")
        self.timestep = int(self.getBasicTimeStep())
        self.map_width = 7
        self.map_height = 7
        self.cell_size = [0.5, 0.5]
        self.grid_origin = [-(self.map_width // 2) * self.cell_size[0], (self.map_height // 2) * self.cell_size[1]]
        self.grid = Grid(self.map_width, self.map_height, self.grid_origin, self.cell_size)
        self.number_of_obstacles = number_of_obstacles
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Get obstacles
        self.obstacles = []
        self.obstacles_starting_positions = []
        obstacles_group = self.getFromDef('OBSTACLES')
        if obstacles_group is not None:
            children_field = obstacles_group.getField('children')
            for i in range(children_field.getCount()):
                obs = children_field.getMFNode(i)
                self.obstacles.append(obs)
                self.obstacles_starting_positions.append(obs.getField('translation').getSFVec3f())
        # Get robot and target
        self.robot = self.getFromDef('robot')
        self.target = self.getFromDef('TARGET')
        # Store their starting positions if needed
        self.robot_starting_position = self.robot.getField('translation').getSFVec3f() if self.robot else None
        self.target_starting_position = self.target.getField('translation').getSFVec3f() if self.target else None
        # Path marker nodes
        self.path_nodes = []
        self.path_nodes_starting_positions = []
        path_group = self.getFromDef('PATH')
        if path_group is not None:
            children_field = path_group.getField('children')
            for i in range(children_field.getCount()):
                node = children_field.getMFNode(i)
                self.path_nodes.append(node)
                self.path_nodes_starting_positions.append(node.getField('translation').getSFVec3f())
        self.path_to_target = []

    def remove_objects(self):
        """Move all obstacles back to their starting positions before placing new ones."""
        for obs, start_pos in zip(self.obstacles, self.obstacles_starting_positions):
            obs.getField('translation').setSFVec3f(start_pos)
            obs.getField('rotation').setSFRotation([0.0, 0.0, 1.0, 0.0])

    def get_grid_coordinates(self, world_x, world_y):
        x = round((world_x - self.grid.origin[0]) / self.grid.cell_size[0])
        y = -round((world_y - self.grid.origin[1]) / self.grid.cell_size[1])
        if self.grid.is_in_range(x, y):
            return x, y
        else:
            return None, None

    def get_random_path(self):
        if not hasattr(self, 'robot_grid_cell') or not hasattr(self, 'target_grid_cell'):
            return None
        print(f"[RandomizerSupervisor] Using robot_grid_cell: {self.robot_grid_cell}, target_grid_cell: {self.target_grid_cell}", flush=True)
        return self.grid.bfs_path(self.robot_grid_cell, self.target_grid_cell)

    def place_path(self, path):
        print("Placing path markers...", flush=True)
        for i, (cell, node) in enumerate(zip(path, self.path_nodes)):
            print(f"Placing marker {i} at cell {cell}", flush=True)
            self.grid.add_cell(cell[0], cell[1], node)

    def reset_path_nodes(self):
        for node, pos in zip(self.path_nodes, self.path_nodes_starting_positions):
            node.getField('translation').setSFVec3f(pos)

    def randomize_obstacles(self):
        max_attempts = 50
        while True:
            self.remove_objects()
            self.grid.empty()
            robot_z = 0.0399261
            available_obstacles = self.obstacles
            selected_obstacles = random.sample(available_obstacles, min(self.number_of_obstacles, len(available_obstacles)))
            placed = 0
            for obs in selected_obstacles:
                if self.grid.add_random(obs):
                    obs.getField('rotation').setSFRotation([0.0, 0.0, 1.0, random.uniform(-np.pi, np.pi)])
                    placed += 1
            # Place robot
            robot_placed = False
            for _ in range(max_attempts):
                robot_cell = self.grid.add_random(self.robot, robot_z)
                if robot_cell:
                    angle = random.uniform(-np.pi, np.pi)
                    self.robot.getField('rotation').setSFRotation([0.0, 0.0, 1.0, angle])
                    robot_placed = True
                    self.robot_grid_cell = robot_cell
                    print(f"[RandomizerSupervisor] Robot grid cell for path: {robot_cell}", flush=True)
                    break
            if not robot_placed:
                continue  # Try a new randomization
            # Try to place target and compute a path
            target_placed = False
            for _ in range(max_attempts):
                target_cell = self.grid.add_random(self.target)
                if target_cell:
                    target_placed = True
                    self.target_grid_cell = target_cell
                    print(f"[RandomizerSupervisor] Target grid cell for path: {target_cell}", flush=True)
                    break
            if not target_placed:
                continue  # Try a new randomization
            # Try to compute a path
            path = self.grid.bfs_path(self.robot_grid_cell, self.target_grid_cell)
            if path is not None:
                self.reset_path_nodes()
                self.path_to_target = path
                print("Computed path:", self.path_to_target, flush=True)
                print("Number of path nodes:", len(self.path_nodes), flush=True)
                if self.path_to_target:
                    self.place_path(self.path_to_target)
                else:
                    print("No path found!", flush=True)
                self.simulationResetPhysics()
                break  # Found a valid path, exit the loop

    def run(self):
        """Main control loop"""
        # print("[RandomizerSupervisor] Starting main control loop")
        
        # Initial randomization
        self.randomize_obstacles()
        self.emitter.send(b"ready")
        # print("[RandomizerSupervisor] Sent initial 'ready' signal on channel", self.emitter.getChannel())
        
        # Setup receiver for reset signals
        self.receiver = self.getDevice('receiver')
        if self.receiver is None:
            # print("[RandomizerSupervisor] Warning: No receiver found!")
            return
        self.receiver.enable(self.timestep)
        self.receiver.setChannel(2)  # Listen on channel 2 for reset signals
        # print(f"[RandomizerSupervisor] Receiver initialized on channel {self.receiver.getChannel()}")
        
        # Main control loop
        while self.step(self.timestep) != -1:
            # Check for reset signal
            if self.receiver.getQueueLength() > 0:
                try:
                    msg = self.receiver.getString()
                    self.receiver.nextPacket()
                    
                    if "reset" in msg:
                        # print("[RandomizerSupervisor] Received reset signal, randomizing environment...")
                        self.randomize_obstacles()
                        self.emitter.send(b"ready")
                        # print("[RandomizerSupervisor] Sent 'ready' signal after randomization")
                except Exception as e:
                    # print(f"[RandomizerSupervisor] Error receiving message: {e}")
                    pass
                continue
            
            # Check if episode is done
            if self.is_episode_done():
                # print("[RandomizerSupervisor] Episode complete, waiting for reset signal...")
                pass

    def is_episode_done(self):
        """Check if current episode is complete"""
        if not hasattr(self, 'robot') or not hasattr(self, 'target'):
            return False
            
        # Get current positions
        robot_pos = self.robot.getPosition()
        target_pos = self.target.getPosition()
        
        # Calculate distance to target
        distance = np.sqrt(sum((robot_pos[i] - target_pos[i])**2 for i in range(3)))
        
        # Check if robot reached target (within 0.075 meters)
        if distance < 0.075:
            # print("[RandomizerSupervisor] Robot reached target!")
            return True
            
        # Check for collision with obstacles
        for obs in self.obstacles[:self.number_of_obstacles]:  # Only check active obstacles
            obs_pos = obs.getPosition()
            dist_to_obs = np.sqrt(sum((robot_pos[i] - obs_pos[i])**2 for i in range(3)))
            if dist_to_obs < 0.015:  # Hard collision threshold
                # print("[RandomizerSupervisor] Robot collided with obstacle!")
                return True
                
        # Check if robot is stuck near obstacles
        if not hasattr(self, 'stuck_near_obstacle_steps'):
            self.stuck_near_obstacle_steps = 0
            
        is_near_obstacle = False
        for obs in self.obstacles[:self.number_of_obstacles]:
            obs_pos = obs.getPosition()
            dist_to_obs = np.sqrt(sum((robot_pos[i] - obs_pos[i])**2 for i in range(3)))
            if dist_to_obs < 0.1:  # Near obstacle threshold
                is_near_obstacle = True
                break
                
        if is_near_obstacle:
            self.stuck_near_obstacle_steps += 1
            if self.stuck_near_obstacle_steps >= 90:  # Same threshold as RL Supervisor
                # print("[RandomizerSupervisor] Robot stuck near obstacles!")
                self.stuck_near_obstacle_steps = 0
                return True
        else:
            self.stuck_near_obstacle_steps = 0
            
        # Check for lack of progress
        if not hasattr(self, 'no_progress_steps'):
            self.no_progress_steps = 0
            self.last_distance = distance
            
        if abs(distance - self.last_distance) < 0.05:
            self.no_progress_steps += 1
            if self.no_progress_steps >= 200:  # Same threshold as RL Supervisor
                # print("[RandomizerSupervisor] Robot not making progress!")
                self.no_progress_steps = 0
                return True
        else:
            self.no_progress_steps = 0
            self.last_distance = distance
            
        return False

if __name__ == "__main__":
    sup = RandomizerSupervisor(number_of_obstacles=10, seed=1)  # Set seed as needed
    sup.run() 