def post_processing(output):
    result ={}
    inx = output["y"].index(max(output["y"]))
    result = {"Y_CLASS":inx}
    return result

