import hpsv2

if __name__=="__main__":
    path = "eval_exp_" # path to directory with images
    hpsv2.evaluate(path, hps_version="v2.1")
