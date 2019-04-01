#%%
def EcbFilename2Tuple(ecb_filename:str) -> tuple:
    # example: ecb_filename="1_12ecbplus.xml"
    # return: (1,12,"ecbplus")
    exact_name = ecb_filename.split(".")[0]
    first_number, second_number_and_type = exact_name.split("_")
    if second_number_and_type.endswith("ecb"):
        second_number, file_type = second_number_and_type.rstrip("ecb"), "ecb"
    else:
        second_number, file_type = second_number_and_type.rstrip("ecbplus"), "ecbplus"
    return (int(first_number), int(second_number), file_type)


def Tuple2EcbFilename(ecb_filename_tuple:tuple) -> str:
    # example: (1,12,"ecbplus")
    # return: ecb_filename="1_12ecbplus.xml"
    return "%s_%s%s.xml" % ecb_filename_tuple