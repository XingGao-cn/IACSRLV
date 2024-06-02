from pathlib import Path
import os

FILE = Path(__file__).resolve()

project_dir = os.path.dirname(FILE)

print(project_dir)

# f = open(project_dir+'\\Var.txt', 'w')
#
#         for i in range():
#             for j in range():
#                 tmp = "{:40.40f}".format()
#                 f.write(str(tmp) + ' ')
#             f.write(" " + '\n')
#         f.write(" " + '\n \n')
#
# f.close()

script_dir = Path(__file__).parent
parent_dir = script_dir.parent
relative_path_to_file = os.path.join(parent_dir, 'Data', 'file.txt')
print(script_dir)

print(parent_dir)

relative_path_to_file = os.path.join(parent_dir, 'Data', 'file.txt')
print(relative_path_to_file)