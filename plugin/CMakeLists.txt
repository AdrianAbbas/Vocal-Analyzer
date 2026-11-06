cmake_minimum_required(VERSION 3.22)
# Vocal Match: the JUCE plug-in (AU + VST3 effect) and VocalMatchRender, the offline tool the
# Python-vs-C++ check uses. Build setup as in AdrianAbbas/LIMINAL (TAPEWORN FX).
# macOS defaults: one universal binary for Apple Silicon and Intel, macOS 10.15+.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Target architectures")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS version")
endif()
project(VocalMatch VERSION 0.2.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(JUCE_PATH "" CACHE PATH "Existing JUCE 8.0.16 source (otherwise fetched)")
if(JUCE_PATH)
    add_subdirectory("${JUCE_PATH}" "${CMAKE_BINARY_DIR}/JUCE" EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(JUCE GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG ea77444750de45e036e0279bb0291144bf8e3688 # JUCE 8.0.16
        GIT_SHALLOW FALSE)
    FetchContent_MakeAvailable(JUCE)
endif()

set(VOCALMATCH_FORMATS VST3)
if(APPLE)
    list(APPEND VOCALMATCH_FORMATS AU)
endif()
# Audio Unit: aufx / Vcmt / Wnwd ("Windward Audio: Vocal Match"). The manufacturer code is
# Windward Audio's (shared with TAPEWORN FX, subtype Tpfx); the subtype is this plug-in's own.
# These codes also make the VST3 class ID: never change them, songs find the plug-in by them.
juce_add_plugin(VocalMatch
    COMPANY_NAME "Windward Audio"
    BUNDLE_ID "com.windwardaudio.vocalmatch"
    PLUGIN_MANUFACTURER_CODE Wnwd
    PLUGIN_CODE Vcmt
    PRODUCT_NAME "Vocal Match"
    PLUGIN_NAME "Vocal Match"
    AU_EXPORT_PREFIX VocalMatchAU
    DESCRIPTION "Your vocal, processed like the reference"
    FORMATS ${VOCALMATCH_FORMATS}
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    VST3_CATEGORIES Fx Dynamics EQ
    AU_MAIN_TYPE kAudioUnitType_Effect
    # Lets sandboxed hosts (GarageBand) load it in-process.
    AU_SANDBOX_SAFE TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE)
target_sources(VocalMatch PRIVATE
    Source/PluginProcessor.cpp Source/PluginEditor.cpp Source/Parameters.cpp Source/PresetFile.cpp)
target_include_directories(VocalMatch PUBLIC Source)
target_compile_definitions(VocalMatch PUBLIC
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0 JUCE_REPORT_APP_USAGE=0)
target_link_libraries(VocalMatch PRIVATE juce::juce_audio_utils
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)

# VocalMatchRender: the plug-in's processor, offline (tests/parity).
juce_add_console_app(VocalMatchRender PRODUCT_NAME "VocalMatchRender")
# The linked plug-in target exports the same setting through its format macro.
get_target_property(vm_render_definitions VocalMatchRender COMPILE_DEFINITIONS)
if(vm_render_definitions)
    list(REMOVE_ITEM vm_render_definitions JUCE_STANDALONE_APPLICATION=1)
    set_target_properties(VocalMatchRender PROPERTIES COMPILE_DEFINITIONS "${vm_render_definitions}")
endif()
target_sources(VocalMatchRender PRIVATE Tools/RenderMain.cpp)
target_link_libraries(VocalMatchRender PRIVATE VocalMatch juce::juce_audio_utils)
