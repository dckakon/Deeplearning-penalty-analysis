import os


# Function to rename multiple files
def main():
    c=0
    for count,filename in enumerate(os.listdir("E:/capstone/newvideo/pic/")):
        dst =  str(c) + ".jpeg"
        src = 'E:/capstone/newvideo/pic/' + filename
        dst = 'E:/capstone/newvideo/pic/' + dst
        c=int(c)+1

        # rename() function will
        # rename all the files
        os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
