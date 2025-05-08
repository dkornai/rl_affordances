import numpy as np
import torch

DIR_DICT = {
    'left': 0,
    'up': 1,
    'right': 2,
    'down': 3
}

class GridElement():
    """
    GridElement objects are the basic building blocks of the grid world. 
    They represent a single cell in the grid and can have walls on their sides.
    Each GridElement can also have neighbors, which are other GridElement objects.
    reward is an optional parameter that can be assigned to each GridElement.
    """
    def __init__(self, x, y, walls=None):
        self.x = x
        self.y = y
        if walls is not None:
            assert len(walls) == 4, "Walls must be a list of four boolean values."
            assert all(wall in [True, False] for wall in walls), "Walls must be True or False."
            self.walls = walls
        else:
            self.walls = [False, False, False, False]  # left, up, right, down

        self.neighbors = [None, None, None, None]  # left, up, right, down

    def add_wall(self, direction):
        if direction in ['left', 'right', 'up', 'down']:
            direction = DIR_DICT[direction]
        if direction in [0, 1, 2, 3]:
            self.walls[direction] = True
        else:
            raise ValueError("Direction must be 0 (left), 1 (up), 2 (right), or 3 (down).")
        
    def add_neighbor(self, direction, neighbor):
        if direction in ['left', 'right', 'up', 'down']:
            direction = DIR_DICT[direction]
        if direction in [0, 1, 2, 3]:
            self.neighbors[direction] = neighbor
        else:
            raise ValueError("Direction must be 0 (left), 1 (up), 2 (right), or 3 (down).")
        
    def get_wall_string(self):
        walls_str = ['x x','   ','x x']
        if self.walls[0]:
            if self.walls[2]:
                walls_str[1] = '| |'
            else:
                walls_str[1] = '|  '
        elif self.walls[2]:
            walls_str[1] = '  |'
        if self.walls[1]:
            walls_str[0] = 'x―x'
        if self.walls[3]:
            walls_str[2] = 'x―x'
        
        return walls_str
    
class Grid():
    """
    A class representing a grid of GridElement objects.
    """

    def __init__(self, grid_elements):
        assert isinstance(grid_elements, list), "grid_elements must be a list of GridElement objects."
        assert all(isinstance(element, GridElement) for element in grid_elements), "All elements must be GridElement objects."
        self.grid_elements = grid_elements
        self.width = max(element.x for element in grid_elements) + 1
        self.height = max(element.y for element in grid_elements) + 1
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for element in grid_elements:
            self.grid[element.y][element.x] = element

    def get_element(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        else:
            raise IndexError("Coordinates out of bounds.")
    
    def __repr__(self):
        """
        Return a string representation of the grid.
        """
        grid_str = ""
        for y in range(self.height-1, -1, -1):
            # Top walls
            for x in range(self.width):
                element = self.get_element(x, y)
                if element is not None:
                    walls_str = element.get_wall_string()
                    # add upper walls to grid_str
                    grid_str += f'{walls_str[0][0:2]}'
                    if x == self.width - 1:  # left boundary
                        grid_str += f'{walls_str[0][2]}\n'
                else:
                    grid_str += '  '

            # Middle walls
            for x in range(self.width):
                element = self.get_element(x, y)
                if element is not None:
                    walls_str = element.get_wall_string()
                    # add middle walls to grid_str
                    grid_str += f'{walls_str[1][0:2]}'
                    if x == self.width - 1:
                        grid_str += f'{walls_str[1][2]}\n'
                else:
                    grid_str += '  '

        # Bottom walls
        for x in range(self.width):
            element = self.get_element(x, 0)
            if element is not None:
                walls_str = element.get_wall_string()
                # add bottom walls to grid_str
                grid_str += f'{walls_str[2][0:2]}'
                if x == self.width - 1:
                    grid_str += f'{walls_str[2][2]}\n'
            else:
                grid_str += '  '
        
        return grid_str

def fullgrid_init(width, height, wall_horiz=None, wall_vert=None):
    """
    Initialize a dense WxH dense grid world 
    The grid is represented as a list of GridElement objects.
    Each GridElement can have walls on its sides and neighbors.

    args:
        width: int, width of the grid
        height: int, height of the grid
        wall_horiz: list of coordinates for horizontal walls
        wall_vert: list of coordinates for vertical walls
    returns:
        grid: Grid object representing the grid world
    """
    assert isinstance(width, int) and width > 0, "Width must be a positive integer."
    assert isinstance(height, int) and height > 0, "Height must be a positive integer."
    assert wall_horiz is None or isinstance(wall_horiz, list), "wall_horiz must be a list of coordinates."
    assert wall_vert is None or isinstance(wall_vert, list), "wall_vert must be a list of coordinates."
    
    # Create a grid of GridElement objects with walls
    grid = []
    for x in range(width):
        for y in range(height):
            element = GridElement(x, y)
            
            # Add walls corresponding to edges of the grid
            if x == 0: # left wall
                element.add_wall('left')
            if x == width - 1: # right wall
                element.add_wall('right')
            if y == 0: # bottom wall
                element.add_wall('down')
            if y == height - 1: # top wall
                element.add_wall('up')
            
            # Add walls based on wall_horiz and wall_vert
            if wall_horiz is not None:
                if [x, y+1] in wall_horiz: # horizontal wall above
                    element.add_wall('up')
                if [x, y] in wall_horiz: # horizontal wall below
                    element.add_wall('down')

            if wall_vert is not None:
                if [x+1, y] in wall_vert: # vertical wall to the right
                    element.add_wall('right')
                if [x, y] in wall_vert: # vertical wall to the left
                    element.add_wall('left')

            grid.append(element)

    grid = Grid(grid)

    # Update neighbors
    for x in range(width):
        for y in range(height):
            element = grid.get_element(x, y)
            
            upper_neighbor = grid.get_element(x, y+1) if y < height - 1 and not element.walls[1] else None
            element.add_neighbor('up', upper_neighbor)
            lower_neighbor = grid.get_element(x, y-1) if y > 0 and not element.walls[3] else None
            element.add_neighbor('down', lower_neighbor)
            left_neighbor = grid.get_element(x-1, y) if x > 0 and not element.walls[0] else None
            element.add_neighbor('left', left_neighbor)
            right_neighbor = grid.get_element(x+1, y) if x < width - 1 and not element.walls[2] else None
            element.add_neighbor('right', right_neighbor)

    return grid

def reassign_nextstate_probs(move_probs, blocked, controllability, wind):
    """
    Apply controllability and wind to the move probabilities.
    - Controllability varies from 0 to 1, where 0 means no control and 1 means full control.\\
      Decreased control increases the probability of staying in the same state.
    - Wind is a vector of two values (x, y) varying from -1 to 1, where 0 means no wind.\\
      Wind affects the probabilities of moving in the direction of the wind.    
      
    args:
        move_probs: numpy array of move probabilities [up, down, left, right, stay]
        blocked: list of booleans indicating if the move is blocked in each direction
        controllability: list of two floats indicating the controllability in x and y directions
        wind: list of two floats indicating the wind in x and y directions
    returns:
        move_probs: numpy array of modified move probabilities [up, down, left, right, stay]
    """
    assert isinstance(move_probs, (np.ndarray)), "move_probs must be a numpy array."
    assert len(move_probs) == 5, "move_probs must be a list of five probabilities."
    assert isinstance(blocked, (list)), "blocked must be a list of booleans."
    assert len(blocked) == 4, "blocked must be a list of four booleans."
    assert isinstance(controllability, (list)), "controllability must be a list of two floats."
    assert len(controllability) == 2, "controllability must be a list of two floats."
    assert isinstance(wind, (list)), "wind must be a list of two floats."
    assert len(wind) == 2, "wind must be a list of two floats."

    control_x = controllability[0]
    assert control_x >= 0 and control_x <= 1, "controllability must be between 0 and 1."
    control_y = controllability[1]
    assert control_y >= 0 and control_y <= 1, "controllability must be between 0 and 1."
    wind_x = wind[0]
    assert wind_x >= -1 and wind_x <= 1, "wind must be between -1 and 1."
    wind_y = wind[1]
    assert wind_y >= -1 and wind_y <= 1, "wind must be between -1 and 1."


    # Add inertia to the agent as a function of controllability
    control_x_loss = (1 - control_x)*sum(move_probs[[0, 2]])
    move_probs[0] = control_x*move_probs[0]
    move_probs[2] = control_x*move_probs[2]
    move_probs[4] += control_x_loss
    control_y_loss = (1 - control_y)*sum(move_probs[[1, 3]])
    move_probs[1] = control_y*move_probs[1]
    move_probs[3] = control_y*move_probs[3]
    move_probs[4] += control_y_loss


    # Add wind to the move probabilities
    if wind_x > 0: # right wind
        # remove energy from the left move and add it to the stay move
        left_loss = (abs(wind_x))*move_probs[0]
        move_probs[0] -= left_loss
        move_probs[4] += left_loss
        # if there is no blockage, move energy from up, down, and stay to the right
        if not blocked[2]:
            right_gain = (abs(wind_x))*sum(move_probs[[1, 3, 4]])
            move_probs[[1, 3, 4]] = (1 - abs(wind_x))*move_probs[[1, 3, 4]]
            move_probs[2] += right_gain
    
    if wind_x < 0: # left wind
        # remove energy from the right move and add it to the stay move
        right_loss = (abs(wind_x))*move_probs[2]
        move_probs[2] -= right_loss
        move_probs[4] += right_loss
        # if there is no blockage, move energy from up, down, and stay to the left
        if not blocked[0]:
            left_gain = (abs(wind_x))*sum(move_probs[[1, 3, 4]])
            move_probs[[1, 3, 4]] = (1 - abs(wind_x))*move_probs[[1, 3, 4]]
            move_probs[0] += left_gain

    if wind_y > 0: # up wind
        # remove energy from the down move and add it to the stay move
        down_loss = (abs(wind_y))*move_probs[3]
        move_probs[3] -= down_loss
        move_probs[4] += down_loss
        # if there is no blockage, move energy from left, right, and stay to the up
        if not blocked[1]:
            up_gain = (abs(wind_y))*sum(move_probs[[0, 2, 4]])
            move_probs[[0, 2, 4]] = (1 - abs(wind_y))*move_probs[[0, 2, 4]]
            move_probs[1] += up_gain
    
    if wind_y < 0: # down wind
        # remove energy from the up move and add it to the stay move
        up_loss = (abs(wind_y))*move_probs[1]
        move_probs[1] -= up_loss
        move_probs[4] += up_loss
        # if there is no blockage, move energy from left, right, and stay to the down
        if not blocked[3]:
            down_gain = (abs(wind_y))*sum(move_probs[[0, 2, 4]])
            move_probs[[0, 2, 4]] = (1 - abs(wind_y))*move_probs[[0, 2, 4]]
            move_probs[3] += down_gain

    return move_probs


class GridWorld():
    """
    A class representing a grid world MDP.
    """

    def __init__(self, width, height, wall_horiz=None, wall_vert=None, controllability=[1.0,1.0], wind=[0.0,0.0]):
        """
        Initialize a grid world with given width and height.
        The grid is represented as a list of GridElement objects.
        Each GridElement can have walls on its sides and neighbors.
        """
        self.width = width
        self.height = height
        self.n_states = width * height
        self.n_actions = 4  # left, up, right, down
        
        self.grid = fullgrid_init(width, height, wall_horiz, wall_vert)
        self.wall_horiz = wall_horiz
        self.wall_vert = wall_vert

        self.controllability = controllability
        self.wind = wind

    def get_TPM(self):
        """
        Get the Transition Probability Matrix (TPM) for the grid world.
        The TPM is a 3D tensor of shape [state, action, next_state].
        """
        tpm = torch.zeros((self.n_states, 4, self.n_states), dtype=torch.float32)
        
        for x_curr in range(self.width):
            for y_curr in range(self.height):
                # Identify the current starting state
                curr_element = self.grid.get_element(x_curr, y_curr)
                curr_index = y_curr * self.width + x_curr
                
                # Get the neighbors of the current element
                neighbour_left = curr_element.neighbors[0]
                neighbour_up = curr_element.neighbors[1]
                neighbour_right = curr_element.neighbors[2]
                neighbour_down = curr_element.neighbors[3]
                neighbours = [neighbour_left, neighbour_up, neighbour_right, neighbour_down]
                target_indices = [neighbour.x + neighbour.y * self.width if neighbour is not None else None for neighbour in neighbours]
                target_indices.append(curr_index)

                # Check if moves are possible in each direction
                blocked_left = True if neighbour_left is None else False
                blocked_up = True if neighbour_up is None else False
                blocked_right = True if neighbour_right is None else False
                blocked_down = True if neighbour_down is None else False
                blocked = [blocked_left, blocked_up, blocked_right, blocked_down]
                
                # Look at the outcomes of possible moves in the four different directions
                for direction in range(4):
                    # Apply blockage to the move probabilities, if required
                    default_move_probs = np.zeros(5)
                    if blocked[direction] is False:
                        default_move_probs[direction] = 1.0
                    else:
                        default_move_probs[4] = 1.0
                    
                    # Apply controllability and wind to the move probabilities
                    move_probs = reassign_nextstate_probs(default_move_probs, blocked, self.controllability, self.wind)
                    
                    # Assign the move probabilities to the TPM
                    for i in range(5):
                        if target_indices[i] is not None:
                            if move_probs[i] > 0:
                                # Assign the move probabilities to the TPM
                                tpm[curr_index, direction, target_indices[i]] = move_probs[i]
                           

        return tpm    
    
    