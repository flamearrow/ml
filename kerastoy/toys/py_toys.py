class WithToy(object):
    def __enter__(self):
        print("entered")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exited")


def main():
    with WithToy() as toy:
        print("doStuff")

    print("doStuffAfterWithBlock")


if __name__ == "__main__":
    main()
