# 0 = empty black rectangle, 1 = dot, 2 = big dot, 3 = vertical line,
# 4 = horizontal line, 5 = top right, 6 = top left, 7 = bot left, 8 = bot right
# 9 = gate

start_multiplayer_1 = [2, 31, 3]
start_multiplayer_2 = [18, 31, 3]

multiplayer_map = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,3,1,3,0,0,0,3,1,3,3,1,3,0,0,0,0,3,1,3,0,0,0,3,1,3,3],
[3,3,1,3,6,4,4,4,8,1,7,4,4,4,8,1,3,3,1,3,6,4,4,4,8,1,7,4,4,4,8,1,3,3],
[3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,3,3,1,6,4,5,1,6,5,1,6,4,4,8,3,1,3,3,1,6,4,5,1,6,5,1,6,4,4,8,3],
[3,3,1,3,3,1,3,0,3,1,3,3,1,3,0,0,0,3,1,3,3,1,3,0,3,1,3,3,1,3,0,0,0,3],
[3,3,1,7,8,1,7,4,8,1,3,3,1,7,4,4,5,3,1,7,8,1,7,4,8,1,3,3,1,7,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,5,1,3,7,4,4,5,1,3,7,4,4,4,4,4,4,5,1,3,7,4,4,5,1,3,3],
[3,6,4,4,4,4,4,4,8,1,7,4,4,4,8,1,3,6,4,4,4,4,4,4,8,1,7,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3],
[3,3,1,7,4,4,4,4,8,1,7,4,4,4,8,1,3,3,1,7,4,4,4,4,8,1,7,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,3],
[3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3],
[3,3,1,7,4,4,4,4,8,1,7,4,4,4,8,1,3,3,1,7,4,4,4,4,8,1,7,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,5,1,6,4,4,4,5,1,3,7,4,4,4,4,4,4,5,1,6,4,4,4,5,1,3,3],
[3,6,4,4,4,4,4,4,8,1,7,4,4,4,8,1,3,6,4,4,4,4,4,4,8,1,7,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,3],
[3,3,1,7,4,4,4,5,3,1,7,4,4,4,8,1,3,3,1,7,4,4,4,5,3,1,7,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,3,3,1,1,1,1,1,1,1,3,3,1,1,1,1,1,3,3,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,5,1,3,3,1,6,5,1,6,4,4,8,7,4,4,4,5,1,3,3,1,6,5,1,6,4,4,8,3],
[3,6,4,4,4,8,1,7,8,1,3,3,1,7,4,4,5,6,4,4,4,8,1,7,8,1,3,3,1,7,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,3],
[3,3,1,6,4,5,1,6,4,4,8,7,4,4,5,1,3,3,1,6,4,5,1,6,4,4,8,7,4,4,5,1,3,3],
[3,3,1,7,4,8,1,7,4,4,4,4,4,4,8,1,3,3,1,7,4,8,1,7,4,4,4,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

start_multiplayer_mirrored_1 = [15, 31, 3]
start_multiplayer_mirrored_2 = [31, 31, 3]

multiplayer_mirrored_map = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,3,1,3,0,0,0,0,3,1,3,3,1,3,0,0,0,3,1,3,0,0,0,0,3,1,3,3],
[3,3,1,7,4,4,4,8,1,7,4,4,4,5,3,1,3,3,1,7,4,4,4,8,1,7,4,4,4,5,3,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,3],
[3,7,4,4,5,1,6,5,1,6,4,5,1,3,3,1,3,7,4,4,5,1,6,5,1,6,4,5,1,3,3,1,3,3],
[3,0,0,0,3,1,3,3,1,3,0,3,1,3,3,1,3,0,0,0,3,1,3,3,1,3,0,3,1,3,3,1,3,3],
[3,6,4,4,8,1,3,3,1,7,4,8,1,7,8,1,3,6,4,4,8,1,3,3,1,7,4,8,1,7,8,1,3,3],
[3,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,8,3,1,6,4,4,4,4,4,4,8,3,1,6,4,4,8,3,1,6,4,4,4,4,4,4,8,3],
[3,3,1,7,4,4,4,8,1,7,4,4,4,4,4,4,5,3,1,7,4,4,4,8,1,7,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3],
[3,3,1,7,4,4,4,8,1,7,4,4,4,4,8,1,3,3,1,7,4,4,4,8,1,7,4,4,4,4,8,1,3,3],
[3,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3],
[3,3,1,7,4,4,4,8,1,7,4,4,4,4,8,1,3,3,1,7,4,4,4,8,1,7,4,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,5,1,6,4,4,4,4,4,4,8,3,1,6,4,4,4,5,1,6,4,4,4,4,4,4,8,3],
[3,3,1,7,4,4,4,8,1,7,4,4,4,4,4,4,5,3,1,7,4,4,4,8,1,7,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,3],
[3,3,1,7,4,4,4,8,1,3,6,4,4,4,8,1,3,3,1,7,4,4,4,8,1,3,6,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,3,3,1,1,1,1,1,3,3,1,1,1,1,1,1,1,3,3,1,1,1,1,1,3,3],
[3,7,4,4,5,1,6,5,1,3,3,1,6,4,4,4,8,7,4,4,5,1,6,5,1,3,3,1,6,4,4,4,8,3],
[3,6,4,4,8,1,3,3,1,7,8,1,7,4,4,4,5,6,4,4,8,1,3,3,1,7,8,1,7,4,4,4,5,3],
[3,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,8,7,4,4,5,1,6,4,5,1,3,3,1,6,4,4,8,7,4,4,5,1,6,4,5,1,3,3],
[3,3,1,7,4,4,4,4,4,4,8,1,7,4,8,1,3,3,1,7,4,4,4,4,4,4,8,1,7,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]


start_singleplayer = [10, 31, 3]

singleplayer_map = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,0,0,0,0,0,0,0,0,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,3,1,3,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,3,6,4,4,4,8,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,3,3,1,6,4,5,1,6,5,1,6,4,4,8,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,8,1,7,4,8,1,3,3,1,7,4,4,5,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,7,4,4,4,4,4,4,5,1,3,7,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,6,4,4,4,4,4,4,8,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,4,8,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,4,8,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,7,4,4,4,4,4,4,5,1,6,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,6,4,4,4,4,4,4,8,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,5,1,6,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,5,3,1,7,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,3,3,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,7,4,4,4,5,1,3,3,1,6,5,1,6,4,4,8,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,6,4,4,4,8,1,7,8,1,3,3,1,7,4,4,5,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,3,3,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,5,1,6,4,4,8,7,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,8,1,7,4,4,4,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

start_singleplayer_mirrored = [23, 31, 3]

singleplayer_mirrored_map = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,0,0,0,0,0,0,0,0,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,3,0,0,0,3,1,3,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,7,4,4,4,5,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,7,4,4,5,1,6,5,1,6,4,5,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,6,4,4,8,1,3,3,1,7,4,8,1,7,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,8,3,1,6,4,4,4,4,4,4,8,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,7,4,4,4,4,4,4,5,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,7,4,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,7,4,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,5,1,6,4,4,4,4,4,4,8,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,7,4,4,4,4,4,4,5,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,4,5,1,6,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,8,1,3,6,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,3,3,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,7,4,4,5,1,6,5,1,3,3,1,6,4,4,4,8,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,6,4,4,8,1,3,3,1,7,8,1,7,4,4,4,5,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,6,4,4,8,7,4,4,5,1,6,4,5,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,7,4,4,4,4,4,4,8,1,7,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

# Games
multiplayer_execution_boards=[multiplayer_map, multiplayer_mirrored_map]
multiplayer_player1_start_execution_positions=[start_multiplayer_1, start_multiplayer_mirrored_1]
multiplayer_player2_start_execution_positions=[start_multiplayer_2, start_multiplayer_mirrored_2]

singleplayer_execution_boards=[singleplayer_map, singleplayer_mirrored_map]
singleplayer_start_execution_positions=[start_singleplayer, start_singleplayer_mirrored]
