# ncpm
non crossing perfect matching

https://en.wikipedia.org/wiki/Catalan_number

## installation

I like using a venv named `env`, but use whatever you want!

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -e .
```

## usage

```bash
$ python3 -m ncpm.draw --help
# start here
```

```bash
$ python3 -m ncpm.draw 2 3 --grid --color rainbow3 --curve formulaic --grid-size 12
# your generated image will open after processing
```

## gallery

![image](gallery/ncpm6.jpg)

![image](gallery/ncpm5.jpg)

![image](gallery/ncpm4.jpg)

![image](gallery/ncpm3.jpg)

![image](gallery/ncpm2.jpg)
