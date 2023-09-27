
function showPage(pageNumber) {
    var pages = document.getElementsByClassName('page-content');
    var temp_class = document.querySelectorAll('ul li a');
    for (var i = 0; i < pages.length; i++) {
        pages[i].style.display = 'none';
    }
    for (var i = 0; i < temp_class.length; i++) {
        temp_class[i].className = '';
    }
    var selectedPage = document.getElementById('page' + pageNumber);
    var selecteda = document.getElementById('a' + pageNumber);
    var dropdown = document.querySelector('.dropdown');
    if (selecteda) {
        selecteda.className = 'active';
    }
    if (selectedPage) {
        selectedPage.style.display = 'block';
    }
    if (pageNumber > 3) {
        dropdown.classList.add('dropdown-show');
    } else {
        dropdown.classList.remove('dropdown-show');
    }
}

$(document).ready(function () {
    $.ajax({
        url: '/get_nickname',
        type: 'GET',
        success: function (data) {
            var nickname = data.nickname;
            document.getElementById('nickname-area').textContent = '欢迎使用: ' + nickname;
            document.getElementById('nickname').textContent = nickname;
            if(data.role===1){
                document.querySelector('.dropdown').style.display = 'none';
            }
            else if(data.role===2){
                document.getElementById('a6').style.display = 'none';
            }
            else{
                document.querySelector('.dropdown').style.display = 'block';
            }
        },
        error: function () {
            console.log('获取用户昵称失败');
        }
    });
    $.ajax({
        url: '/count_m_p',
        type: 'GET',
        success: function (data) {
            var mcnt = data.model_count;
            var pcnt = data.prediction_count;
            document.getElementById('m_cnt').textContent = mcnt;
            document.getElementById('p_cnt').textContent = pcnt;
        },
        error: function () {
            console.log('请求失败');
        }
    });
    setInterval(function () {
        $.ajax({
            url: '/count_m_p',
            type: 'GET',
            success: function (data) {
                var mcnt = data.model_count;
                var pcnt = data.prediction_count;
                document.getElementById('m_cnt').textContent = mcnt;
                document.getElementById('p_cnt').textContent = pcnt;
            },
            error: function () {
                console.log('请求失败');
            }
        });
    }, 4000);
    $("#logout-link").click(function () {
        $.ajax({
            url: '/logout',
            type: 'GET',
            success: function (data) {
                if (data.status == 'success') {
                    window.location.href = "/login"; // 重定向到登录页面
                } else {
                    console.log("登出失败");
                }
            },
            error: function () {
                console.log('请求登出接口失败');
            }
        });
    });
    $.ajax({
        url: '/avg_count',
        type: 'GET',
        dataType: 'json',
        success: function (data) {
            var newData = [];
            for (var i = 0; i < 6; i++) {
                newData.push(['类别' + i, data['class' + i + '_count']]);
            }
            Highcharts.chart('zs', {
                chart: {
                    type: 'pie',
                    options3d: {
                        enabled: true,
                        alpha: 45
                    }
                },
                title: {
                    text: '平均验证结果展示：',
                    align: 'left'
                },
                subtitle: {
                    text: '--3D饼状图--',
                    align: 'left'
                },
                plotOptions: {
                    pie: {
                        innerSize: 100,
                        depth: 45
                    }
                },
                series: [{
                    name: '个数',
                    data: newData
                }],
                credits: {
                    enabled: false
                }
            });
        },
    });
    setInterval(function () {
        $.ajax({
        url: '/avg_count',
        type: 'GET',
        dataType: 'json',
        success: function (data) {
            var newData = [];
            for (var i = 0; i < 6; i++) {
                newData.push(['类别' + i, data['class' + i + '_count']]);
            }
            Highcharts.chart('zs', {
                chart: {
                    type: 'pie',
                    options3d: {
                        enabled: true,
                        alpha: 45
                    }
                },
                title: {
                    text: '平均验证结果展示：',
                    align: 'left'
                },
                subtitle: {
                    text: '--3D饼状图--',
                    align: 'left'
                },
                plotOptions: {
                    pie: {
                        innerSize: 100,
                        depth: 45
                    }
                },
                series: [{
                    name: '个数',
                    data: newData
                }],
                credits: {
                    enabled: false
                }
            });
        },
    });
    }, 7000);
})
window.onload = function() {
    fetchAnnouncement();
}

function fetchAnnouncement() {
    fetch('/announcement', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        if (data.content && data.content !== "No announcement available") {
            displayAnnouncement(data.content);
        }
    })
    .catch(error => {
        console.error('Error fetching announcement:', error);
    });
}

function displayAnnouncement(content) {
    const announcementBox = document.getElementById('announcement-box');
    const announcementContent = document.getElementById('announcement-content');
    const overlay = document.getElementById('overlay');

    announcementContent.textContent = content;

    overlay.classList.remove('hidden');  // 显示遮罩
    announcementBox.classList.remove('hidden');  // 显示公告框
}

function closeAnnouncement() {
    const announcementBox = document.getElementById('announcement-box');
    const overlay = document.getElementById('overlay');

    announcementBox.classList.add('hidden');
    overlay.classList.add('hidden');
}
new Vue({
    el: '#avatar',

});
