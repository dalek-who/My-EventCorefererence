import os

file_dir = "../../test/t1/"
file = "t.txt"
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
with open(os.path.join(file_dir+file), "w") as f:
    f.write("hello world!")
    print((os.path.join(file_dir+file)))
