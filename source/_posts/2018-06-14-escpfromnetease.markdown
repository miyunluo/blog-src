---
layout:     post
title:      "50行解码网易云音乐歌曲"
tags:
    - fun
---

> “网易云音乐劝退文”

## 前言

那是和往常一样平静的一天，我像往常一样打开网易云开始下载歌曲，然后像往常一样打算放进MP3里，然后...等等，这是什么...

相信很多人应该也发现了，网易云新推出了一种加密格式ncm，这下只能使用网易云播放器不说，即使你下载了歌曲，过了会员期也照样不能再播放了… ( 有句话不知当讲不当讲

其实很多用户充会员的目的无非就是为了下载高品质的音乐，虽说8元每月的会员费也谈不上很昂贵，但这种强迫用户续费的逻辑着实让人大跌眼镜。这个时候知乎上必定又会有一部分人站出来说什么版权意识还需加强云云，照这个道理，岂不是买本书也需要加密了，而且不得相互借阅？虽说亚马逊确实推出了新的电子书加密格式，但是电子书属于一次买断，永久阅读，且价格合理，大多数用户还是愿意为此买单的。且不说网易云的做法完全不考虑MP3用户的感受，会员制听音乐的思路让我看不到任何诚意。

---

## Setup

1. 找到缓存文件

ncm格式目前还没有解码方法，这里采取另一种思路，从缓存文件解码获得源音乐文件。一般来讲音乐播放软件为了保证播放流畅性，都会一边播放一边缓存。网易云音乐的音乐缓存文件后缀名为 `.uc!`，直接在文件管理器中搜索  `.uc!` 就可以找到文件位置。缓存文件的名字比较长，比如

`78284-_-_320-_-_68ad2b2606e66f43232ee877563ab513.uc!`

这里 `78284` 是歌曲id，id可以从歌曲url获得，这样根据歌曲id就可以定位对应歌曲的缓存文件了。

2. 解码

解码的方法很简单，缓存文件每个字节和0xA3进行或操作就可以了...方法来源于网络，但是出处找不到了-_-||。程序默认解码后文件后缀名是mp3，如果缓存文件是flac，将后缀名改为.flac。

```python
import os
import sys

KEY = 0xA3
def decode(src_path, dest_path):
    try:
        fin = open(src_path, "rb")
    except IOError as e:
        print(str(e))
        return
    try:
        fout = open(dest_path, "wb")
    except IOError as e:
        print(str(e))
        return

    song_encode = fin.read()
    song_decode = bytearray()
    for i, byte in enumerate(song_encode):
        sys.stdout.write("\r处理进度: %d%%" % (round((i + 1) * 100 / len(song_encode))))
        sys.stdout.flush()
        if type(byte) == str: #python 2
            song_decode.append(int(byte.encode("hex"), 16) ^ KEY)
        else:                 #python 3
            song_decode.append(byte ^ KEY)
    
    print()
    fout.write(song_decode)
    fin.close()
    fout.close()

def main():
    if len(sys.argv) !=2:
       print("使用 python uc!decoder.py [source]")
    else:
        last = sys.argv[1].rfind(os.path.sep)
        src_path = sys.argv[1][:last + 1]
        dest_path = sys.argv[1][:last + sys.argv[1][last:].find(".")] + ".mp3"
        print("Source path: %s\nDestination path: %s" % (sys.argv[1], dest_path))
        decode(sys.argv[1], dest_path)
        print("如果缓存为无损歌曲，请将后缀名由mp3改为flac")

if __name__ == '__main__':
    main()
```

**网易云音乐今后有可能改变缓存文件的加密方式，因此该方法具有时效性，并不长期有效。**

## Result

从 `uc!` 文件获得了《君の名は》专辑的flac文件。

![img](/images/in-post/post-blog-kimino.png)

## Src

[github](https://github.com/miyunluo/MusicDownloader/tree/master/Netease_uc!Decode)