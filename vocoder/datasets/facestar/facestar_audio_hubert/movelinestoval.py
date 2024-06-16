with open('train.txt', 'r') as f:
    lines = f.readlines()

with open('val.txt', 'a') as f:
    f.writelines(lines[200:400])  # Change 2:5 to the lines you want to move

with open('train.txt', 'w') as f:
    f.writelines(lines[:200] + lines[400:])  # Change 2 and 5 to the lines you want to keep

