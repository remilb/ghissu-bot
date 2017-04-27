import os

def scrape(filename):
    with open(filename) as file:
        print(filename)
        for line in file:
            print (line)
    pass
def main():
    filename = os.getcwd() + '/sample/S01E01.txt'
    scrape(filename)

main()
