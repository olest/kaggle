# file I/O
def read_csv(file_path, has_header = True):
    cnt = 0
    with open(file_path) as f:
        if has_header: f.readline()
        data = []
        for line in f:
            cnt += 1
            line = line.strip().split(",")
            data.append([float(x) for x in line])
        print "Read ",cnt," lines";
        return data

def write_csv(file_path, data):
    with open(file_path,"w") as f:
        for line in data: f.write(",".join(line) + "\n")

def write_delimited_file(file_path,data,header=None, delimiter=","):
    f_out = open(file_path,"w")
    if header is not None:
        f_out.write(delimiter.join(header) + "\n")
        for line in data:
            if isinstance(line, str):
                f_out.write(line + "\n")
            else:
                f_out.write(delimiter.join(line) + "\n")
        f_out.close()

