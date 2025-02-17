// 加载店铺链接
load $店铺链接

// 等待5秒
delay 5

// 获取店铺名称
$店铺名称 = (function() {
    var t = document.querySelector("#crumb-wrap > div > div.contact.fr.clearfix.shieldShopInfo").innerText
        .replace(/联系客服/g, '')
        .replace(/关注店铺/g, '');

    if (t == '') {
        t = document.querySelector("#crumb-wrap > div > div.contact.fr.clearfix.shieldShopInfo > div.J-hove-wrap.EDropdown.fr > div:nth-child(1) > div").innerText
            .replace(/联系客服/g, '')
            .replace(/关注店铺/g, '');
    }

    return t;
})();

// 等待3秒
delay 3