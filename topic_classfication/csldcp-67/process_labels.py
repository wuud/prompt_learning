# 用于将标签转化为两个字的特殊标签，方便做mask language model相关任务
label_des2tag={
"材料科学与工程":"材料",
"作物学":"作物",
"口腔医学":"口腔",
"药学":"药学",
"教育学":"教育",
"水利工程":"水利",
"理论经济学":"理经",
"食品科学与工程":"食品",
"畜牧学/兽医学":"兽医",
"体育学":"体育",
"核科学与技术":"核能",
"力学":"力学",
"园艺学":"园艺",
"水产":"水产",
"法学":"法学",
"地质学/地质资源与地质工程":"地质",
"石油与天然气工程":"能源",
"农林经济管理":"农林",
"信息与通信工程":"通信",
"图书馆、情报与档案管理":"情报",
"政治学":"政治",
"电气工程":"电气",
"海洋科学":"海洋",
"民族学":"民族",
"航空宇航科学与技术":"航空",
"化学/化学工程与技术":"化工",
"哲学":"哲学",
"公共卫生与预防医学":"卫生",
"艺术学":"艺术",
"农业工程":"农工",
"船舶与海洋工程":"船舶",
"计算机科学与技术":"计科",
"冶金工程":"冶金",
"交通运输工程":"交通",
"动力工程及工程热物理":"动力",
"纺织科学与工程":"纺织",
"建筑学":"建筑",
"环境科学与工程":"环境",
"公共管理":"公管",
"数学":"数学",
"物理学":"物理",
"林学/林业工程":"林业",
"心理学":"心理",
"历史学":"历史",
"工商管理":"工商",
"应用经济学":"应经",
"中医学/中药学":"中医",
"天文学":"天文",
"机械工程":"机械",
"土木工程":"土木",
"光学工程":"光学",
"地理学":"地理",
"农业资源利用":"农资",
"生物学/生物科学与工程":"生物",
"兵器科学与技术":"兵器",
"矿业工程":"矿业",
"大气科学":"大气",
"基础医学/临床医学":"医学",
"电子科学与技术":"电子",
"测绘科学与技术":"测绘",
"控制科学与工程":"控制",
"军事学":"军事",
"中国语言文学":"语言",
"新闻传播学":"新闻",
"社会学":"社会",
"地球物理学":"地球",
"植物保护":"植物"
}

label_des2tag_reverse={v:k for k,v in label_des2tag.items()}
label_twoword_list=[v for k,v in label_des2tag.items()]
print([k for k,v in label_des2tag.items()])
print("label_twoword_list:",label_twoword_list,len(label_twoword_list))

# labeltwoword2index_dict={x:index for index,x in enumerate(label_twoword_list)}
# print("labeltwoword2index_dict:",labeltwoword2index_dict)

label2index_dict={label_des2tag_reverse[x]:index for index,x in enumerate(label_twoword_list)}
print("label2index_dict:",label2index_dict,len(label2index_dict))

all_core_words = []
for item in label_des2tag.items():
    core_words = []
    k = item[0]
    v = item[1]
    core_words.append(k)
    # core_words.append(v)
    all_core_words.append(core_words)
print(all_core_words)
