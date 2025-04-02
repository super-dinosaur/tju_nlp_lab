from icecream import ic
import re

if __name__ == "__main__":
    s1 = "B  IO OBIO  BIII IIIII"
    # construct s2 which replace one or more spaces with one space
    s2 = re.sub(r'\s+', ' ', s1)
    ic(s1,s2)
