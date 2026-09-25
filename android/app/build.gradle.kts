// The app: the library from `cargo ndk` under `src/main/jniLibs`, the
// manifest, and a signature. Nothing is compiled here.
plugins {
    id("com.android.application")
}

// A release build is signed with the key in these variables when they are
// set, and with Gradle's own debug key when they are not, so a build always
// gives an APK that installs. The release workflow sets them from the
// repository's secrets; see the README.
val keystore = System.getenv("ANDROID_KEYSTORE_FILE")

android {
    namespace = "com.playforge.sandy3"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.playforge.sandy3"
        // Vulkan 1.1 and a GPU with compute shaders are a given from Android
        // 9 on, which is also as far back as the Rust target reaches.
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }

    if (keystore != null) {
        signingConfigs {
            create("release") {
                storeFile = file(keystore)
                storePassword = System.getenv("ANDROID_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("ANDROID_KEY_ALIAS")
                keyPassword = System.getenv("ANDROID_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = if (keystore != null) {
                signingConfigs.getByName("release")
            } else {
                signingConfigs.getByName("debug")
            }
        }
    }

    packaging {
        jniLibs {
            // The library is already stripped and built once per ABI, so it
            // is packed as it is rather than compressed and re-extracted.
            useLegacyPackaging = false
        }
    }
}
