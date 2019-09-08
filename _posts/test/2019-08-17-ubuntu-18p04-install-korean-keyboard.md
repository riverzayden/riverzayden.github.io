---
layout: post
title: 우분투(Ubuntu) 18.04 한글 키보드 설치
category: Linux
tag: [리눅스 설정, Ubuntu]
---
# iBus 기반 한글 키보드 설치

Ubuntu 18.04 LTS 버전 기준으로 한글 키보드를 설치하는 방법입니다.
Ubuntu 18.04에서 한글어 입력기는 다시 `ibus`가 되었기 때문에 14.04나 16.04 때와는 다릅니다.

* 메뉴에서 `Language Support` 실행 → 필요한 파일들 자동으로 설치 됨
* 메뉴에서 `Region & Language` 실행
* `Input Sources` 항목에서 기본으로 잡혀있던 `English`는 삭제하고, `Korean(Hangul)` 선택
* 아래에 있는 설정 버튼 클릭
* `Hangul Toggle Key`의 `Add` 버튼을 누르고 <kbd>한글</kbd> 키 입력(ALT_R 로 표시될 것임)

<br>

# UIM 기반 한글 설정

## 한글 설정

Ubuntu 18.04 LTS의 한글 입력기는 `iBus`로 되어 있습니다. 하지만, Sublime Text나 Visual Studio Code 등의 프로그램에서 한글 입력이 되지 않는 문제가 있어서 방법을 찾아보니 `UIM`을 이용하면 상당부분 해소가 되는 것 같았습니다. 물론 `UIM`도 완벽하지는 않은 것 같습니다. Visual Studio Code에서 `간` 등의 글자 입력이 잘 안되는 경우가 있네요. Sublime이나 IntelliJ 등에서는 문제없이 동작하네요.

설치는 다음과 같습니다.

<pre class="prettyprint">
sudo apt install uim
</pre>

<br>

* `Settings > Region & Language`에서 `Manage Installed Language` 버튼 클릭.
* 입력기를 `UIM`으로 변경
* 재부팅

재부팅 후, 프로그램 메뉴에서 `Input Method` 실행(`UIM`이라고 타이핑해도 실행됩니다.)

* `Global Settings`에서 `Specify default IM` 체크
* `Global Settings > Default Input method`를 `Byeoru`로 선택
* `Toolbar`의 `Display`를 `Never`로 설정
* `Global key bindings 1`의 상단의 `[Global] on`과 `[Global] off` 항목을 빈 칸으로 설정
* `BVyeoru key bindings 1`의 `[Byeoru] on`과 `[Byeoru] off` 키 설정을 `Multi_key`로 설정







#### 출처

https://snowdeer.github.io/linux/2018/07/11/ubuntu-18p04-install-korean-keyboard/

