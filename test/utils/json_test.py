import json

dc = {"loss_each_batch": {1:0.1, 2:0.2, 3:0.3}}
with open("./t.json", "w") as f:
    json.dump(dc, f)
