---
layout:     post
title:      "50行实现定时发送天气预报提醒邮件"
tags:
    - fun
---

> “想不想每天给她/他发送一条天气提醒呢”


## 起因

1. 魔都的天气，太魔性了...尤其是梅雨季
2. ios自带天气app无提醒，经常忘记查看。使用过 Mr.Weather、Colorful Pro，提醒又过多，导致通知栏里全是天气通知
3. 需求其实很简单，一份简介明了又不扰人的天气提醒
4. 使用邮件无需额外费用，比短信提醒易于实现，使用node.js更是十分简单

---

## Setup

使用node.js实现天气获取与邮件发送，首先需要安装node。

+ macOS 推荐使用 homebrew 进行安装，也可下载Installer安装

  ```shell
  >$ brew install node
  ```

+ Ubuntu

  ```Shell
  >$ curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
  >$ sudo apt-get install -y nodejs
  ```

+ Windows

  直接下载Installer https://nodejs.org/en/download

## 注册Heweather

获取天气信息需要依赖提供天气查询服务的网站。在对比了和风天气与丫丫天气后，发现同为免费用户的情况下，和风天气提供的信息更为全面，这里使用和风。(至今没收到丫丫的验证邮件...)

+ 用户注册

  地址：http://www.heweather.com

+ 获得想要查询城市或区域的编码：

  地址：https://www.heweather.com/documents/city

  如 松江区：CN101020900，闵行区：CN101020200。

+ 获得认证key

  注册后登录，地址：控制台->我的控制台->认证key

+ 免费查询api

  根据和风天气文档，获取天气信息大合集的url如下，返回一份json数据，具体字段信息请参考官方文档

  https://free-api.heweather.com/s6/weather?location=CITY_CODE&key=YOUR_KEY

  将 **CITY_CODE** 替换为第二步得到的城市代码，如CN101020900

  将 **YOUR_KEY** 替换为第三步自己的key

## 开启SMTP

使用smtp协议进行邮件发送，需要开启邮箱的smtp服务，一般在邮箱的设置中。

以163邮箱为例，smtp在 设置->POP3/SMTP/IMAP 选项下。

![img](/images/in-post/post-blog-POP3SMTPIMAP.png)

初次开启，会要求设置授权码，此授权码就是之后使用smtp协议发邮件时的密码，注意与邮箱登录密码区别。

![img](/images/in-post/post-blog-AuthCode.png)

QQ邮箱开启方法类似。

---

## 问题分解

1. **请求并处理天气数据**

首先当然是要获得需要的天气数据，由于要进行网络请求，这里使用 `request` 模块。

```javascript
const request=require('request');
const url= 'https://free-api.heweather.com/s6/weather?location=***********&key=********************************';
request(url,(error,response,body)=>{};
```

返回参数 `body` 包含我们需要的天气信息，部分信息格式化后的内容如下。

```json
{
    "HeWeather6": [
        {
            "basic": {
                "cid": "CN101020900", 
                "location": "松江", 
                "parent_city": "上海", 
                "admin_area": "上海", 
                "cnty": "中国", 
                "lat": "31.03046989", 
                "lon": "121.22354126", 
                "tz": "+8.00"
            }, 
            "status": "ok", 
            "now": {
                "cloud": "0", 
                "cond_code": "101", 
                "cond_txt": "多云", 
                "fl": "38", 
                "hum": "56", 
                "pcpn": "0.0", 
                "pres": "1003", 
                "tmp": "34", 
                "vis": "10", 
                "wind_deg": "123", 
                "wind_dir": "东南风", 
                "wind_sc": "2", 
                "wind_spd": "6"
            }, 
            "daily_forecast": [
                {
                    "cond_code_d": "100", 
                    "cond_code_n": "101", 
                    "cond_txt_d": "晴", 
                    "cond_txt_n": "多云", 
                    "date": "2018-05-16", 
                    "hum": "74", 
                    "mr": "05:40", 
                    "ms": "19:34", 
                    "pcpn": "0.0", 
                    "pop": "0", 
                    "pres": "1005", 
                    "sr": "04:59", 
                    "ss": "18:44", 
                    "tmp_max": "36", 
                    "tmp_min": "25", 
                    "uv_index": "10", 
                    "vis": "13", 
                    "wind_deg": "181", 
                    "wind_dir": "南风", 
                    "wind_sc": "4-5", 
                    "wind_spd": "25"
                }, 
                
            ], 
            ......
```

当然根据和风天气的文档，已知这是一个json文件，直接使用json parser对 `body` 处理即可。

```javascript
data=JSON.parse(body);
data=data.HeWeather6[0];
```

json中包含的信息比较多，可以筛选一些重要的信息来使用。比如 日出时间: `daliy_forecast.sr`，日落时间: `daliy_forecast.ss`，最高温度: `daliy_forecast.tmp_max`，最低温度: `daliy_forecast.tmp_min`，降水概率: `daliy_forecast.pop` 等。

2. **邮件编排**

获得了天气数据后，就可以发送邮件了，但是如何编辑排列得到的天气数据也是一个需要考虑的问题，发一堆杂乱无章的数据，视觉观感不好。当然邮件本身支持html样式，可以编写一个简单的html样式来排列邮件内容。

最终进行邮件发送时，只需要将这一样式表拼接为字符串即可。

```html
<div>
   <div>
    <h1>给主席的天气预报</h1>
   </div>
   <div>
    <h3 style="margin:20px auto 10px auto"> data.daily_forecast[0].date   data.basic.location 区</h3>
   <div>
     <p>
     	<span>日出时间: </span>
     	<span> data.daily_forecast[0].sr </span>
     	<span style="width:40%;margin-left: 30px">日落时间: </span>
     	<span> data.daily_forecast[0].ss+ </span>
     </p>
     <p>
     	<span>白天: </span>
     	<span> data.daily_forecast[0].cond_txt_d </span>
     	<span style="width:40%;margin-left: 30px">晚间: </span>
     	<span> data.daily_forecast[0].cond_txt_n </span>
     </p>
     <p>
     	<span>最高温度: </span>
     	<span><b>data.daily_forecast[0].tmp_max</b>℃</span>
     	<span style="width:40%;margin-left: 30px">最低温度: </span>
     	<span><b>data.daily_forecast[0].tmp_min</b>℃</span>
     	</p>
     <p>
     	<span>紫外线强度指数: </span>
     	<span><b>data.daily_forecast[0].uv_index</b></span>
     </p>
     <p>
     	<span>降水概率: </span>
     	<span>data.daily_forecast[0].pop %</span>
     </p>
     <p>
     	<span>风力: </span>
     	<span> data.daily_forecast[0].wind_sc </span>
     </p>
     <p>
     	<span>穿衣建议: </span>
     	<span> data.lifestyle[1].txt </span>
     </p>
     <br>
     <p>
     	<small>注 紫外线强度: 0-2 无危险 | 3-5 较轻伤害 | 6-7 很大伤害 | 8-10 极高伤害 | 11+ 及其危险</small>
     </p>
    </div>
   </div>
</div>
```

3. **邮件发送**

发送邮件使用 `nodemailer` 模块。

```javascript
const nodemailer=require('nodemailer');
// 建立SMTP连接
var smtpTransport = nodemailer.createTransport({
  host: "smtp.163.com", // smtp主机
  secureConnection: true, // 使用SSL
  port: 465, // SMTP 端口
  auth: {
    user: "邮箱前缀@163.com", // 账号
    pass: "******" // 授权码
  }
});
 
// 设置邮件内容
var mailOptions = {
  from: "名字 <邮箱前缀@163.com>", // 发件地址
  to: "XXXXXXX@example.com", // 收件地址
  subject: "Hello", // 标题
  html: "<b>邮件由nodemailer发送</b>" // html实例，替换为第2步的天气内容
}
 
// 发送邮件
smtpTransport.sendMail(mailOptions, function(error, response){
  if(error){
    console.log(error);
  }else{
    console.log("Message sent: " + response.message);
  }
  smtpTransport.close(); // 关闭连接
});
```

4. **定时**

Unix 与 Linux本身提供了定时启动指令 `crontab`，当然 node.js 也有自己的定时器，而且使用起来更加方便。

使用模块 `node-schedule`。

```javascript
const schedule=require('node-schedule');
var rule = new schedule.RecurrenceRule();
rule.hour = 6; rule.minute = 30; rule.second = 0; // 定时 06:30:00
var run = schedule.scheduleJob(rule, YOUR_FUNC); // YOUR_FUNC 为想要执行的函数
```

至此所有模块已经基本完成，之后将天气查询，与邮件发送功能封装在一个函数中，传入 `schudle.scheduleJob` 即可。

## 完整代码

```javascript
const request = require('request');
const nodemailer = require('nodemailer');
const schedule = require('node-schedule');
var rule = new schedule.RecurrenceRule();
rule.hour = 6; rule.minute = 30; rule.second = 0; // 设置定时发送的时间
var data= '';
var j = schedule.scheduleJob(rule, function(){
	console.log('发送天气预报...');
  	sendWeather();
});
function sendWeather(){
	const url= 'https://free-api.heweather.com/s6/weather/forecast?location=城市代码&key=自己的key'; // 城市代码: 从和风天气文档获得，key: 注册和风天气得到
	request(url,(error,response,body)=>{
		if(error){
			console.log(error);
		}
		data=JSON.parse(body);
		data=data.HeWeather6[0]

		let transporter=nodemailer.createTransport({
		host: "smtp.163.com", // 163邮箱的smtp服务器地址
		port: 465,
		secureConnection: true,
		auth: {
			user:'邮箱前缀@163.com', // 邮箱地址
			pass:'*******' // smtp授权码，不是邮箱的登录密码
			}
		});
		let mailOptions={
			from: "预报君 <邮箱前缀@163.com>", // 与发件地址一致，否则报错
			to: "to someone @foxmail.com", // 要发送的邮箱地址
			subject: data.daily_forecast[0].date + 'の天気',
			text:'城市 :'+data.basic.location+'时间：'+ data.daily_forecast[0].date,
			html: '<div><div><h1>给主席的天气预报</h1></div><div><h3 style="margin:20px auto 10px auto">' + data.daily_forecast[0].date + ' ' + data.basic.location + '区</h3><div><p><span>日出时间: </span><span>'+data.daily_forecast[0].sr+'</span><span style="width:40%;margin-left: 30px">日落时间: </span><span>'+data.daily_forecast[0].ss+'</span></p><p><span>白天: </span><span>'+data.daily_forecast[0].cond_txt_d+'</span><span style="width:40%;margin-left: 30px">晚间: </span><span>'+data.daily_forecast[0].cond_txt_n+'</span></p><p><span>最高温度: </span><span><b>'+data.daily_forecast[0].tmp_max+'</b>℃</span><span style="width:40%;margin-left: 30px">最低温度: </span><span><b>'+data.daily_forecast[0].tmp_min+'</b>℃</span></p><p><span>紫外线强度指数: </span><span><b>'+data.daily_forecast[0].uv_index+'</b></span></p><p><span>降水概率: </span><span>'+data.daily_forecast[0].pop+' %</span></p><p><span>风力: </span><span>'+data.daily_forecast[0].wind_sc+'</span></p><p><span>穿衣建议: </span><span>' + data.lifestyle[1].txt + '</span></p><br><p><small>注 紫外线强度: 0-2 无危险 | 3-5 较轻伤害 | 6-7 很大伤害 | 8-10 极高伤害 | 11+ 及其危险</small></p></div></div></div>'
		};
		transporter.sendMail(mailOptions,(err,info)=>{
			if(err){
				return console.log(err);
			}else{
				console.log('Message %s sent %s',info.messageId,info.response);
			}
			transporter.close();
		});
	});
}
```

## 最终效果

![img](/images/in-post/post-blog-WeatherEmail.jpg)

## 最后

只需要不到50行就可以自动发送天气邮件了，你是不是也想给她/他每天发一条天气提醒呢？不妨试一试。[github](https://github.com/miyunluo/Funny_Tools/tree/master/Weather_Email_Reporter)

