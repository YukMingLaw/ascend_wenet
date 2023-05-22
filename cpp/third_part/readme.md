```
tar -xvf openfst-1.63.tar.gz
cd openfst-1.6.3
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/fst ..
make -j32 && make install
```

```
unzip kenlm.zip
cd kenlm
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/kenlm  -DBUILD_SHARED_LIBS=ON ..
make -j32 && make install
```

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/fst/lib:/usr/local/kenlm/lib
```