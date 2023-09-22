# 0 = empty black rectangle, 1 = dot, 2 = big dot, 3 = vertical line,
# 4 = horizontal line, 5 = top right, 6 = top left, 7 = bot left, 8 = bot right
# 9 = gate
original_board = [
[6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
[3, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 2, 3, 0, 0, 3, 1, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 1, 3, 0, 0, 3, 2, 3, 3],
[3, 3, 1, 7, 4, 4, 8, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 7, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 7, 4, 4, 4, 4, 5, 1, 3, 7, 4, 4, 5, 0, 3, 3, 0, 6, 4, 4, 8, 3, 1, 6, 4, 4, 4, 4, 8, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 6, 4, 4, 8, 0, 7, 8, 0, 7, 4, 4, 5, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[8, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 9, 9, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 7],
[4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4],
[5, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 7, 4, 4, 4, 4, 4, 4, 8, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 6],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 4, 4, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 6, 4, 4, 4, 4, 8, 1, 7, 8, 0, 7, 4, 4, 5, 6, 4, 4, 8, 0, 7, 8, 1, 7, 4, 4, 4, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 5, 3, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 3, 6, 4, 8, 1, 3, 3],
[3, 3, 2, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 3, 3],
[3, 7, 4, 5, 1, 3, 3, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 3, 3, 1, 6, 4, 8, 3],
[3, 6, 4, 8, 1, 7, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 8, 1, 7, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 4, 4, 8, 7, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 8, 7, 4, 4, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 3],
[7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8]
         ]

black_line = [
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,0,0],
]

maze_A = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,5,0,0,6,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,3,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,5,1,3,0,0,3,1,6,4,4,4,4,4,4,5,1,3,3],
[3,3,2,3,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,2,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,6,4,4,4,4,4,4,4,8,1,3,0,0,3,1,3,0,6,4,4,4,4,8,1,3,3],
[3,3,1,3,3,1,1,1,1,1,1,1,1,1,3,0,0,3,1,3,0,3,1,1,1,1,1,1,3,3],
[3,3,1,3,3,1,6,4,4,4,4,4,4,4,8,0,0,3,1,3,0,3,1,6,4,4,4,4,8,3],
[3,3,1,3,3,1,3,0,0,6,4,4,4,4,4,4,4,8,1,3,0,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,1,1,1,1,1,1,1,1,3,0,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,6,4,4,4,4,4,4,4,8,0,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,3,0,0,0,0,0,0,6,4,5,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,3,1,3,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3],
[3,3,1,3,3,1,7,4,4,8,1,3,6,4,4,4,4,4,8,1,3,3,1,7,4,4,4,4,5,3],
[3,3,1,3,3,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,3,3],
[3,3,1,3,7,4,4,4,4,4,4,8,3,1,6,4,4,4,4,4,8,7,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,6,4,4,4,4,4,4,4,5,0,0,3,1,3,3],
[3,3,2,3,0,0,0,0,0,0,0,0,3,1,3,3,1,1,1,1,1,1,1,3,0,0,3,2,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,7,4,4,4,4,4,5,1,3,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,3,1,3,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,3,1,3,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,3,1,3,0,0,3,1,3,3],
[3,3,1,7,4,4,4,4,4,4,4,4,8,1,3,0,0,0,0,0,0,3,1,7,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,3,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,7,4,4,4,4,4,4,8,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8]
]

maze_B = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,5,0,0,0,0,0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3],
[3,3,1,3,0,0,0,0,0,0,0,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,6,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,6,4,4,4,4,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,3,1,6,4,4,4,4,4,4,4,4,4,4,4,8,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,7,4,4,4,4,4,4,4,8,1,3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,8,7,4,4,4,4,4,4,4,4,4,4,4,5,1,3,3],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,0,0,0,0,0,3,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,6,4,5,3,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,3,1,3,3],
[3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,8,1,3,3,1,7,4,4,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,7,4,4,4,4,4,4,4,4,4,8,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

maze_C = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,6,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,7,4,4,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3,1,7,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,7,4,4,4,4,4,4,4,4,4,4,5,0,3,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,0,7,4,4,4,4,4,4,4,4,4,4,4,8,3],
[3,7,4,4,4,4,4,4,4,4,4,4,5,1,3,6,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,6,4,4,4,4,4,4,4,4,5,1,3,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,6,4,4,4,4,4,4,4,4,4,4,8,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,4,4,8,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,7,4,8,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

maze_D = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,5,6,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,5,1,3,3,1,6,4,4,4,4,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,6,4,4,4,4,4,4,4,8,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,3,1,1,1,1,1,1,1,1,1,3,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,7,4,4,4,4,4,4,4,4,4,8,3,1,3,6,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,7,4,4,4,4,4,4,4,4,4,4,5,3,1,3,3,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,3,1,6,4,4,4,4,4,4,4,8,3],
[3,7,4,4,4,4,4,4,4,4,4,4,5,1,3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,3],
[3,6,4,4,4,4,4,4,4,4,4,4,8,1,3,3,1,3,3,1,7,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,3,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,4,4,8,3,1,3,7,4,4,4,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,7,4,8,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

maze_E = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,3],
[3,3,1,3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3,1,3,3],
[3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,3,3],
[3,3,1,3,3,1,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,3,1,3,3],
[3,3,1,3,3,1,3,6,4,4,4,4,4,4,4,4,4,5,0,0,0,0,0,3,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,3,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,6,4,4,4,4,4,5,1,3,0,0,0,0,0,3,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,3,0,0,0,0,0,3,1,3,0,0,0,0,0,3,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,3,0,0,0,0,0,3,1,3,0,0,0,0,0,3,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,3,0,0,0,0,0,3,1,3,0,0,0,0,0,3,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,3,0,0,0,0,0,3,1,3,0,0,0,0,0,3,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,3,0,0,0,0,0,3,1,7,4,4,4,4,4,8,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,3,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,3,3,1,3,3],
[3,3,1,3,3,1,3,3,1,3,0,0,0,0,0,7,4,4,4,4,4,4,4,4,4,8,3,1,3,3],
[3,3,1,3,3,1,3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,3,3,1,3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,3],
[3,3,1,3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,0,0,0,0,0,0,3],
[3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,3],
[3,3,1,3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,0,3],
[3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,4,4,5,0,0,3,1,3,0,0,0,0,0,0,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,3,1,3,0,0,0,0,0,0,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,0,0,3,1,3,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,3],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,1,3,0,0,0,0,0,0,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

maze_F = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,3,3,1,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,3],
[3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,3,1,3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3],
[3,3,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,3],
[3,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,4,4,4,4,4,4,4,8,1,3,3],
[3,3,1,7,4,4,4,4,4,4,4,4,4,4,4,4,4,5,3,1,1,1,1,1,1,1,1,1,3,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,1,6,4,4,4,4,4,4,4,8,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,3],
[3,6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,1,3,0,0,0,0,0,0,0,0,3],
[3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,3],
[3,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]

path_1 = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,0,0,0,0,6,4,4,4,4,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,6,4,5,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,7,4,4,8,1,7,4,4,4,4,5,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,4,4,4,5,1,6,4,4,4,4,8,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,5,1,3,0,0,0,0,0,0,0,7,4,8,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,4,4,4,4,4,5,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,4,4,4,5,1,6,4,4,4,4,8,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,7,4,8,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,6,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,7,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,4,4,4,4,4,5,1,6,4,4,4,4,8,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,7,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,4,8,0,0,0,0,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]
path_2 = [
[6,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,8,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,4,4,4,4,4,8,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,4,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,6,4,4,4,4,8,1,7,4,4,4,4,5,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,1,1,1,1,1,1,1,1,1,1,3,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,6,4,4,4,4,4,4,3,4,4,8,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,6,4,4,4,4,8,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,3,1,1,1,1,1,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,7,4,4,4,4,5,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[3,0,0,0,0,0,0,0,0,0,7,4,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
[7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8],
]