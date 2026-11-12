
#define WINVER 0x0601
#define _WIN32_WINNT 0x0601

#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>

// ===============================
// 日志保存路径
// ===============================

const char* LOG_DIR =
    "D:\\PC_Log";

const char* LOG_FILE =
    "D:\\PC_Log\\power_log.txt";

// ===============================
// 获取当前系统时间
// ===============================

std::string FormatLocalDateTime()
{
    SYSTEMTIME st;
    GetLocalTime(&st);

    const char* weekdays[] =
    {
        "星期日",
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六"
    };

    std::ostringstream oss;

    oss << st.wYear << "-"
        << std::setw(2) << std::setfill('0')
        << st.wMonth << "-"
        << std::setw(2)
        << st.wDay << " "
        << weekdays[st.wDayOfWeek] << " "
        << std::setw(2) << st.wHour << ":"
        << std::setw(2) << st.wMinute << ":"
        << std::setw(2) << st.wSecond;

    return oss.str();
}

// ===============================
// 写入日志文件
// ===============================

bool WriteLog(const std::string& message)
{
    // 创建日志目录
    CreateDirectoryA(LOG_DIR, NULL);

    // 判断文件是否存在
    DWORD attr = GetFileAttributesA(LOG_FILE);
    bool newFile = (attr == INVALID_FILE_ATTRIBUTES);

    std::ofstream file(
        LOG_FILE,
        std::ios::binary | std::ios::app
    );

    if (!file.is_open())
        return false;

    // 新文件写入 UTF-8 BOM
    if (newFile)
    {
        const unsigned char bom[] =
        {
            0xEF, 0xBB, 0xBF
        };

        file.write(
            reinterpret_cast<const char*>(bom),
            3
        );
    }

    file << FormatLocalDateTime()
         << " "
         << message
         << "\r\n";

    file.close();

    return !file.fail();
}

// ===============================
// 关闭 Windows 防火墙
// ===============================

bool DisableFirewall()
{
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));

    si.cb = sizeof(si);

    char commandLine[] =
        "netsh.exe advfirewall set allprofiles state off";

    BOOL result = CreateProcessA(
        "C:\\Windows\\System32\\netsh.exe",
        commandLine,
        NULL,
        NULL,
        FALSE,
        CREATE_NO_WINDOW,
        NULL,
        "C:\\Windows\\System32",
        &si,
        &pi
    );

    if (!result)
        return false;

    // 等待命令执行完成
    DWORD waitResult = WaitForSingleObject(
        pi.hProcess,
        20000
    );

    DWORD exitCode = 1;

    if (waitResult == WAIT_OBJECT_0)
    {
        GetExitCodeProcess(
            pi.hProcess,
            &exitCode
        );
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return waitResult == WAIT_OBJECT_0
        && exitCode == 0;
}

// ===============================
// 程序入口
// ===============================

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        WriteLog("程序启动失败：缺少运行参数");
        return 1;
    }

    std::string mode = argv[1];

    // ---------------------------
    // 开机
    // ---------------------------

    if (mode == "startup")
    {
        bool firewallOK = DisableFirewall();

        if (firewallOK)
        {
            WriteLog("开机（防火墙关闭成功）");
        }
        else
        {
            WriteLog("开机（防火墙关闭失败）");
        }
    }

    // ---------------------------
    // 正常关机或重启
    // ---------------------------

    else if (mode == "shutdown")
    {
        WriteLog("关机");
    }

    // ---------------------------
    // 意外关机
    // ---------------------------

    else if (mode == "unexpected")
    {
        WriteLog(
            "异常关机（6008事件检测时间，非实际断电时刻）"
        );
    }

    else
    {
        WriteLog("程序启动失败：未知运行参数");
        return 1;
    }

    return 0;
}
