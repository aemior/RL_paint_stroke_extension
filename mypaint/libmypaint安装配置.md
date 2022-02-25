# libmypaint 安装配置

## 0. 安装环境依赖包

**系统的环境包**

centos
```
sudo yum install -y gcc gobject-introspection-devel json-c-devel glib2-devel
sudo yum install -y git python autoconf intltool gettext libtool
sudo yum builddep libmypaint
```

ubuntu
```
sudo apt install -y build-essential
sudo apt install -y libjson-c-dev libgirepository1.0-dev libglib2.0-dev
sudo apt install -y python autotools-dev intltool gettext libtool
sudo apt build-dep mypaint # will get additional deps for MyPaint (GUI)
sudo apt build-dep libmypaint  # may not exist; included in mypaint
```
ubuntu-2
```
apt-get install --no-install-recommends -y git wget unzip build-essential libjson-c-dev libgirepository1.0-dev libglib2.0-dev intltool
apt-get install --no-install-recommends -y python-gi-dev liblcms2-dev libgtk-3-dev swig
```

**conda的环境包**

```
conda install -y jsonpatch matplotlib
conda install -y -c conda-forge pygobject visdom
conda install -y -c pkgw/label/superseded gtk3 librsvg
pip install opencv-python==4.5.1.48
```

## 1.解压文件

```shell
tar -xvf ./mypaint-2.0.1.tar.xz -C ./
tar -xvf ./libmypaint-1.6.1.tar.xz -C ./
tar -xvf ./mypaint-brushes-2.0.2.tar.xz -C ./

mv ./mypaint-2.0.1 ./mypaint
mv ./libmypaint-1.6.1 ./libmypaint
mv ./mypaint-brushes-2.0.2 ./mypaint-brushes
```

## 2. 建立资源文件夹

- ```
  mkdir resource
  ```

## 3. 安装libmypaint

- ```shell
  cd ./libmypaint
  ```

- ```shell
  ./configure --prefix=刚刚建立resource文件夹的绝对路径
  ```

- ```shell
  make install
  ```
  
- ```shell
  export  LD_LIBRARY_PATH=刚刚建立resource文件夹的绝对路径/lib:$LD_LIBRARY_PATH
  ```
  
- ```shell
  export PKG_CONFIG_PATH=刚刚建立resource文件夹的绝对路径/lib/pkgconfig:$PKG_CONFIG_PATH
  ```


## 4. 安装mypaint-brush

- ```shell
  cd ../mypaint-brushes
  ```

- ```shell
  ./configure --prefix=刚刚建立resource文件夹的绝对路径
  ```

- ```shell
  make && make install
  ```

- ```shell
  export PKG_CONFIG_PATH=刚刚建立resource文件夹的绝对路径/share/pkgconfig:$PKG_CONFIG_PATH
  ```
## 5. 安装mypaint

- ```
  ldconfig #更新链接库缓存
  ```

- ```shell
  cd ../mypaint
  ```

- ```shell
  python setup.py demo
  ```

- ```shell
  cp build/lib.linux-x86_64-3.*/lib/_mypaintlib.*.so ./lib/_mypaintlib.so
  ```

## 6.改bashrc

```
export  LD_LIBRARY_PATH=刚刚建立resource文件夹的绝对路径/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=刚刚建立resource文件夹的绝对路径/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=刚刚建立resource文件夹的绝对路径/share/pkgconfig:$PKG_CONFIG_PATH
```

