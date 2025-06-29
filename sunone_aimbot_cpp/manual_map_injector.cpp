void RunBotMain()
{
    SetDllDirectoryA("C:\\Users\\%USERNAME%\\AppData\\Local\\AimRuntime\\libs");
    SetCurrentDirectoryA("C:\\Users\\%USERNAME%\\AppData\\Local\\AimRuntime");

    try {
        // ... ВСЯ ТВОЯ ЛОГИКА ИЗ main() ...
    }
    catch (const std::exception& e) {
        std::cerr << "[DLL] Exception: " << e.what() << std::endl;
    }
}
