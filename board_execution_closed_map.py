# 0 = empty black rectangle, 1 = dot, 2 = big dot, 3 = vertical line,
# 4 = horizontal line, 5 = top right, 6 = top left, 7 = bot left, 8 = bot right
# 9 = gate

closed_map_horizontal = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,6,4,4,4,4,4,4,4,5,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,1,1,2,1,1,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,1,1,2,1,1,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,7,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

closed_map_vertical = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,6,4,4,4,4,4,4,4,5,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,2,3,0,0,0,3,2,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,7,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

# 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN

start_closed_map_right_player_1 = [12, 19, 3] # [x, y, direction to start]
start_closed_map_right_player_2 = [18, 19, 3] # [x, y, direction to start]
desired_directions_right = [0, 1] # [player1, player2]

start_closed_map_left_player_1 = [18, 19, 3]
start_closed_map_left_player_2 = [12, 19, 3]
desired_directions_left = [1, 0] # [player1, player2]

start_closed_map_up_player_1 = [12, 19, 1]
start_closed_map_up_player_2 = [12, 13, 1]
desired_directions_up = [2, 3] # [player1, player2]

start_closed_map_down_player_1 = [12, 13, 1]
start_closed_map_down_player_2 = [12, 19, 1]
desired_directions_down = [3, 2] # [player1, player2]

closed_map_boards=[closed_map_horizontal, closed_map_horizontal, closed_map_vertical, closed_map_vertical] * 6
closed_map_player1_positions=[start_closed_map_right_player_1, start_closed_map_left_player_1, start_closed_map_up_player_1, start_closed_map_down_player_1] * 6
closed_map_player2_positions=[start_closed_map_right_player_2, start_closed_map_left_player_2, start_closed_map_up_player_2, start_closed_map_down_player_2] * 6
desired_directions = [desired_directions_right, desired_directions_left, desired_directions_up, desired_directions_down] * 6
