// 清空输入框

//使用 document.querySelector 通过 CSS 选择器找到输入框元素。focus() 和 click() 确保输入框处于活动状态。
//通过设置 value = '' 清空输入框。创建一个 input 事件并触发，确保页面能够感知到输入框的变化。
(function() {
    let inputElement = document.querySelector("body > app-root > layout-default > section > app-kuaidiguoji > nz-card > div > osharp-ad-search > div > span:nth-child(5) > div > nz-input-group > input");
    inputElement.focus(); // 聚焦输入框
    inputElement.click(); // 点击输入框
    inputElement.value = ''; // 清空输入框的值
    let event = new Event('input', { bubbles: true }); // 创建一个 input 事件
    inputElement.dispatchEvent(event); // 触发 input 事件
})();

// 延迟3秒
delay(3);

// 赋值到输入框

//找到输入框元素并设置其值为 $快递单号（这里 $快递单号 可能是变量或占位符）。触发 input 事件，确保页面能够感知到输入框的变化。
(function() {
    let inputElement = document.querySelector("body > app-root > layout-default > section > app-kuaidiguoji > nz-card > div > osharp-ad-search > div > span:nth-child(5) > div > nz-input-group > input");
    inputElement.value = '$快递单号'; // 将快递单号填入输入框
    let event = new Event('input', { bubbles: true }); // 创建一个 input 事件
    inputElement.dispatchEvent(event); // 触发 input 事件
})();

// 延迟3秒
delay(3);

// 执行搜索
//使用 document.querySelector 找到搜索按钮并触发 click() 事件
document.querySelector("body > app-root > layout-default > section > app-kuaidiguoji > nz-card > div > osharp-ad-search > div > span:nth-child(6) > nz-button-group > button:nth-child(1)").click();

// 延迟5秒
delay(5);

// 取得查询结果
//使用 document.querySelectorAll 找到所有结果行。遍历每一行，提取文本内容并拼接成字符串。最终结果存储在 $搜索结果中.
$搜索结果 = (function() {
    var lines = '';
    let rows = document.querySelectorAll("body > app-root > layout-default > section > app-kuaidiguoji > nz-card > div > st > nz-table > nz-spin > div > div > nz-table-inner-scroll > div.ant-table-body.ng-star-inserted > table > tbody > tr");
    for (let i = 0; i < rows.length; i++) {
        let txt = rows[i].querySelector("td.text-left.ant-table-cell.ng-star-inserted");
        if (txt) {
            lines = lines ? lines + ',' + txt.innerText : txt.innerText;
        }
    }
    return lines;
})();

// 延迟2秒
delay(2);

// 条件判断
//如果包含，设置 $比对结果 为 "Y"，并播放声音。如果不包含，设置 $比对结果 为 "N"。注意：Contains 和 SoundPlayer 是 C# 语法，这里可能是伪代码。
$比对结果 = (function() {
    var res = "";
    if ("$搜索结果".Contains("$快递单号")) {
        res = "Y";
        string audioFilePath = @"C:\Windows\Media\notify.wav";
        using (SoundPlayer player = new SoundPlayer(audioFilePath)) {
            player.PlaySync();
        }
    } else {
        res = "N";
    }
    return res;
})();

// 延迟2秒
delay(2);

// 移除匹配上的记录
//remove 不是有效的 JavaScript 语法，可能是伪代码。
if ("$比对结果" == "Y") {
    remove "$比对结果";
}

// 退出条件
//exit 不是有效的 JavaScript 语法，可能是伪代码。
if (1 == 2) {
    exit;
}

// 输出结果
//如果 $比对结果 为 "N"，输出快递单号、比对结果和查询结果。否则，输出 "11"。
output("$比对结果" == "N" ? "$快递单号 $比对结果 $搜索结果" : "11");

// 延迟2秒
delay(2);