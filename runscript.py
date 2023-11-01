import subprocess

for i in range(7):
    subprocess.call(['python', 'fishtracking_distance.py', str(i)])