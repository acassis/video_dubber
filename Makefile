CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -Wno-format-truncation
LDLIBS  = -lm

dubber: dubber.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f dubber

.PHONY: clean
