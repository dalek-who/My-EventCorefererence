from utils.DrawMetricsPicture import result_visualize
import json

path = "./train_curve_datas.json"
with open(path, "r") as f:
    result = json.load(f)

output_path = "./pictures"
result_visualize(result, output_dir=output_path)
