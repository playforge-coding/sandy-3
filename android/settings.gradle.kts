// The Gradle project that turns the shared library `cargo ndk` builds into an
// APK. There is no Java or Kotlin in it: the app is the system's
// NativeActivity loading that library, as the manifest under `app` says.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "sandy-3"
include(":app")
