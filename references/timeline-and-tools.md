# 时间轴、通用工具与合成

## 环境和资料

脚本依赖 Python 3、Pillow、NumPy 与 PATH 中可用的 ffmpeg/ffprobe；项目登记和帧映射主体使用标准库。先用各工具的 --help 探测参数。优先复用 agent 工作区已配置的运行时，不假定某台机器的绝对路径或安装包管理器。字体选择与缺字检查以 render_brand.py 的帮助为准，公开示例不附商业字体。

语音优先用录课工具的逐词缓存/已有字幕。若需ASR，探测已有whisper-cli与本地中文模型；用CPU、有限线程、分段上下文，先识别真实音频。不把某一环境的动态库补丁、平台特定路径或权限设置无条件套用到其他机器。未有转写能力，继续稿件整理和素材登记，明确该步骤未完成，不能捏造“听到”或时间戳。

PDF中的引用图应提取并核对页面/原图完整性；文稿和导图按格式读取，需要时应用可用PDF或文档技能。不要把整个PDF截图当作每个引用图，遗漏跨页内容。

## 课次目录

`project.py init`接受--project、--title、--mode theory|practical|mixed，以及可重复的--narration/--script/--operation/--image/--video和--bgm。登记绝对路径、SHA、角色、真实ffprobe信息；原片只读，不复制人物到skill。BGM 必须显式提供；未提供时不登记假想曲目。显式文件缺失时按工具报告处理，不能假定已经配上。--project已含project.json时拒绝覆盖；修订新建版本目录。

初始化同时保存`project-style.json`和project.json中的profile/style_file：theory沿理论布局，practical/mixed 使用实操布局；新课默认音乐7%/口播3.5%。本课明确选择可用`--style-overrides <JSON>`递归覆盖。实际合成器、图示布局与混音使用本课配置，不回落到旧的左下实操或10%配乐；仅修订已有课程音频时保留该课已确认值，不用新课默认覆盖，详见 [audio-cleanup.md](audio-cleanup.md)。各CLI支持的配置范围不同，按下节接口调用，不能给所有渲染模式盲传同一参数。工具是制作组件，仍须实际编排、渲染和审查。

课次目录包含private、transcript、edit、assets、render、qa、delivery。同素材重复输入归并；不同路径同内容按hash识别。`project.py audit --project DIR`只做结构和素材审计，镜头未准备时NOT_READY，不等于最终媒体PASS。

## 三个时钟

- 原口播时钟：source narration，未经剪辑。
- 原录屏时钟：每个operation素材独立计时；同录制也要确认源内时差。
- 成片时钟：去重/重录剪好、插入章节间隙后的统一输出时钟。

视频时钟默认适配到60fps，记录真实源fps。区间一律左闭右开、整数帧。音频48k每帧800样本。录屏加速段的输出长度等于源长度/倍率（统一量化），原声持续原速；禁止重新取整累计导致长课漂移。

## 编辑工具

以下 CLI 示例从技能包根目录运行，使用 scripts/ 前缀定位工具。输入和输出名称是参数示例；实际制作时替换为独立项目目录内的完整路径，不把原片、时间线或成片写入已安装的技能包。

```
python3 scripts/timeline.py build-map --frames N --cuts cuts.json --out edit-map.json
python3 scripts/timeline.py map-captions --map edit-map.json --captions transcript-source.json --out captions-edited.json --srt captions.srt
python3 scripts/timeline.py cut-pcm --map edit-map.json --input source-48k.wav --out narration-edited.wav --ramp-ms 4
```

N是本课原口播60fps参考总帧数，不复制其他课程数值。cuts必须有start_frame/end_frame/reason/evidence；已确认无词的淡变边界额外声明`ramp_safe:{start:true,end:true}`。普通cut证据不自动授权ramp。默认保护原输入与已有输出。

source字幕是列表，包含start_frame/end_frame/text及可选id。映射工具保留交集并记录clip/omit/跨cut源间隙；agent核对是否合并同源cue和恢复自然断句，不能将软件重复文本误当讲师再次说话。章卡补时是另一个明确inserts阶段，工具未做就不能声称做了；须对人声、人物、字幕、操作锚点统一重新映射，原cuts不再重复执行。

## 镜头记录

在课次根保存shots.json：

```json
{
  "fps":60,"frames":840,
  "shots":[
    {"id":"s01","kind":"operation","start_frame":0,"end_frame":504,"asset_id":"operation-001","source":{"asset_id":"operation-001","start_frame":0,"end_frame":504},"presenter_visible":true,"subtitle_visible":true,"objective":"验证操作结果","focus_cues":[]},
    {"id":"guard-out","kind":"operation","start_frame":504,"end_frame":522,"source":{"asset_id":"operation-001","start_frame":504,"end_frame":522},"presenter_visible":false,"subtitle_visible":false,"content_fade":"out"},
    {"id":"s02","kind":"transition","start_frame":522,"end_frame":600,"presenter_visible":false,"subtitle_visible":false,"title":"下一步"},
    {"id":"guard-in","kind":"operation","start_frame":600,"end_frame":618,"source":{"asset_id":"operation-001","start_frame":522,"end_frame":540},"presenter_visible":false,"subtitle_visible":false,"content_fade":"in"},
    {"id":"s03","kind":"operation","start_frame":618,"end_frame":840,"source":{"asset_id":"operation-001","start_frame":540,"end_frame":762},"presenter_visible":true,"subtitle_visible":true,"objective":"下一步骤","focus_cues":[]}
  ],
  "transitions":[{"start_frame":522,"end_frame":600,"presenter_visible":false,"subtitle_visible":false}],
  "image_usage":[]
}
```

示例仅说明核心与两侧各18帧净空的时间线关系；`content_fade`由本课合成器实际实现，不是登记工具自动渲染的效果。真实语音必须在保护区之前结束、之后再开始；不足时按本课EDL补无词停顿，不能直接照示例切掉口播。

asset_id替换为本课manifest里的ID。source区间以适配60fps后的源片段帧计时，保留实际源时间/倍率信息于alignment记录；未标单位的start/end不能混用。真实结果停帧可写source:{asset_id,hold_frame}，冻结帧必须在已登记视频长度内，不填伪造的移动区间。kind为presenter/cover/diagram/figure/operation/montage/process/reading/transition。reading用于完整原文或指定停读，显式记录presenter_visible、subtitle_visible及恢复边界。montage可以有clips:[{source:{asset_id,start_frame,end_frame},...}]；使用crop/speed也不能把重复源区间变新素材。image_usage记录asset_id、shot_id及完整全图位置。

每shot记录objective、对应口播cue、实操所属应用/对象/控件、instruction/action/result或仅menu/state的证据类型、源区间、加速/保持/删段、布局、讲师与字幕状态。zoom/focus默认空，使用时写明具体教学原因。新课chapter数量与时长从本课定义读取；源素材不足不是用空白片、无关视频或假点击补齐的理由。

## 画面与字幕工具

```
python3 scripts/render_brand.py --cover --title '本课标题' --subtitle '本课目标' --speaker 讲师 --out cover.png
python3 scripts/render_brand.py --diagram assets/diagram-practical-example.json --project-style project-style.json --highlight verify --out diagram.png
python3 scripts/render_brand.py --caption '按实际口播生成的句子' --caption-style dark --out subtitle.png
python3 scripts/render_brand.py --transition --title '章节标题' --out-dir transition-frames
python3 scripts/level_narration.py --input narration-edited.wav --out narration-leveled.wav --report qa/narration-level.json --target-lufs -18 --max-true-peak -2
python3 scripts/mix_audio.py --narration narration-leveled.wav --bgm music-for-mix.wav --out audio --project-style project-style.json --fps 60 --gain-trace
```

混音前先执行 [气声与底噪处理](audio-cleanup.md) 和 [响度校准](audio-loudness.md)。示例目标来自当前系列配置，用户覆盖时替换为本课值。`level_narration.py`保护已有输出，输入输出采样数不一致或实测目标不合格时不发布WAV；查看报告后再混音。没有 BGM 时直接将校准后的人声母版用于最终合成；mix_audio.py 的 --bgm 是必填参数，不为无配乐项目伪造曲目。`music-for-mix.wav` 代表本课已检查的音源：需要清理时使用处理后的浮点WAV，已经干净时直接传实际原曲路径。`mix_audio.py`保持传入校准母版不变并负责循环/闪避/混合，本身不会自动执行降噪或响度统一，不能仅调用它就宣称完成这些步骤。

图示spec明确nodes的位置/关系、edge line或arrow与锚点，不让渲染器发明教学关系。--highlight只改变强调不改变结构。`--project-style`读取本课配置；没有项目文件时可用`render_brand.py --profile practical`选实操布局，默认理论布局不能直接套实操。转场78张PNG为全RGB不透明，首尾浅底净空；不要在它上方再加头像或字幕。合成器在核心外各18帧保护区处理内容淡化与人像隐藏/回场，确保首末不闪人像。报告按实际实现描述人像硬隐藏或淡变。

`render_brand.py`的`--profile`和`--project-style`当前只用于`--diagram`模式，封面/字幕条/章节 PNG 使用通用固定模板，不传这两个选项。外部合成器仍按project-style落实讲师位置、完整隐私遮罩和保护区；生成一组章卡PNG不代表已完成转场合成。

两个示例spec分属theory（`assets/diagram-example.json`）和practical（`assets/diagram-practical-example.json`），均只是虚构关系模板；按本课语义和所选profile重排节点，不把理论模板坐标直接传给较窄实操面板。当前 render_brand.py 的 --project-style 只给 diagram 应用本课布局；颜色等其余参数支持范围以实际实现和 --help 为准，字体按可用的显式选项设置；若本课明确改其它品牌参数，需在实际合成器落实并验证，不能仅改JSON后声称画面已改变。

实际口播源先判断是否全画幅、已有圆窗或已合成讲师。全画幅的圆窗裁切需本课看帧/人脸位置校准，不复用其他讲师或素材的人脸裁切坐标；已合成小窗不能再叠一次。录屏的菜单栏、联系人、账号或token如与教学无关，裁切或遮盖并核对不损失操作目标。

字幕轨可用render_subtitle API生成透明2560×160，合成位置y=1280，按cue精确frame持续。空白cue和末尾覆盖完整输出帧，实测轨总帧数；不要假定 ffmpeg 一定包含 drawtext/ASS，先探测当前构建。静态图示可以用代码/PIL/SVG，不调用生成图片模型代替可编辑逻辑图。

## 最终合成

按本课记录显式落实每个镜头的 presenter_visible 和隐藏理由，不能仅按 figure/cover 类型自动隐藏。先独立渲染每个有教学意义的 shot（图示、原图重点、操作加速/zoom、结果保持、讲师与镜头衔接），实测每shot帧数，再按同一输出序列拼接。chapter core保持78帧并在讲师/字幕合成阶段禁用这两层；普通背景交叠维持一条讲师时间轴。

ffmpeg concat来源色彩/SPS不同的片段时，最终输入使用`-reinit_filter 0`，否则可能中途重置字幕overlay导致截断。`-threads 2`和filter_threads2控制本机压力。H.264 编码参数来自课次配置，CPU libx264 可作为可移植路径；硬件编码仅在环境支持且用户录制不受影响时考虑。保持标准尺寸/帧率并记录实际编码器。最终叠字幕、映射独立mixed音轨、章节元数据，固定`-frames:v <本课frames>`、AAC48k双声道256k、faststart。MP4 可显式设置 -movie_timescale 48000，避免默认毫秒级编辑列表截短非整毫秒的有效音频尾部；该参数不代替最终有效采样数与包时间轴实测。

先导出临时文件，确认视频帧数、音轨时长、同步和视觉后改为交付名。不要覆盖source或上一版link；禁止把已含章节补时的母版再应用原 cuts/inserts。仅结构工具audit通过不能作为最终片通过。

最终媒体可运行：

```
python3 scripts/verify_media.py --final delivery/course.mp4 --shots shots.json --reference-audio audio/mixed.wav --captions captions-edited.json --chapters chapters.json --project-style project-style.json --out qa/final-media.json
```

新课必须传`--project-style`，使检查器读取实际圆窗、字幕带与保护区范围；无配置仅能检查旧理论左下核心，并明确返回REVIEW，不能放行实操新标准。返回TECHNICAL_PASS仅表示报告列明的真实媒体/同步/章卡数值检查完成；教学语义、可读性、嘴型和宏观统一仍需按delivery-and-qa执行。脚本从本课shots动态读取总帧数，不使用其他课程的固定总帧数。
