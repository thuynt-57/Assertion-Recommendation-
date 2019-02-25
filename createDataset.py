import re

input_path = "/home/s1710414/MachineLearning/minorResearch/assertionGenerator/libsignal-protocol-c-tokens.txt"
output_path = "/home/s1710414/MachineLearning/minorResearch/assertionGenerator/libsignal-protocol-c-data.csv"
output_file = open(output_path,'w')

output = "line, mem_access, signed_overflow, unsigned_overflow, shift, index_bound, division_by_zero, float_to_int\n"
flag_mem_access = "0"
flag_signed_overflow = "0"
flag_unsigned_overflow = "0"
flag_shift = "0"
flag_index_bound = "0"
flag_division_by_zero = "0"
flag_float_to_int = "0"
flag_comment = "0"

with open(input_path) as input_file:
    for l in input_file:
        if bool(re.search("rte", l)) and bool(re.search("mem_access", l)):
            flag_mem_access = "1"
        elif bool(re.search("rte", l)) and bool(re.search("unsigned_overflow", l)):
            flag_unsigned_overflow = "1"
        elif bool(re.search("rte", l)) and bool(re.search("signed_overflow", l)):
            flag_signed_overflow = "1"
        elif bool(re.search("rte", l)) and bool(re.search("shift", l)):
            flag_shift = "1"
        elif bool(re.search("rte", l)) and bool(re.search("index_bound", l)):
            flag_index_bound = "1"
        elif bool(re.search("rte", l)) and bool(re.search("division_by_zero", l)):
            flag_division_by_zero = "1"
        elif bool(re.search("rte", l)) and bool(re.search("float_to_int", l)):
            flag_float_to_int = "1"
        elif l.startswith("/"):
            flag_comment = "1"
        else:
            output += re.sub(r'[\n\r]+', '', l) + ", " +  flag_mem_access + ", " +  flag_signed_overflow + ", " + flag_unsigned_overflow + ", " +  flag_shift + ", " +  flag_index_bound + ", " + flag_division_by_zero + ", " +  flag_float_to_int + "\n"
            flag_mem_access = "0"
            flag_signed_overflow = "0"
            flag_unsigned_overflow = "0"
            flag_shift = "0"
            flag_index_bound = "0"
            flag_division_by_zero = "0"
            flag_float_to_int = "0"
            flag_comment = "0"

output_file.write(output)

input_file.close()
output_file.close()