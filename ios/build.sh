#!/bin/sh
# Build Sandy 3 for iOS and put it in an app bundle.
#
#   ios/build.sh sim       build for the simulator, install it on the booted
#                          one (booting an iPhone if none is) and launch it,
#                          with the log on this terminal
#   ios/build.sh device    build for a phone and sign the bundle, which then
#                          installs with `xcrun devicectl device install app`
#
# The app is the same binary a desktop runs, built for the iOS target and
# wrapped in a folder with `Info.plist`. There is no Xcode project. A device
# build needs a development signing identity and a provisioning profile for
# com.playforge.sandy3, given as IOS_SIGNING_IDENTITY and
# IOS_PROVISIONING_PROFILE (a path to the .mobileprovision file).
set -eu
cd "$(dirname "$0")/.."

kind=${1:-sim}
case "$kind" in
    sim)
        target=aarch64-apple-ios-sim
        platform=iPhoneSimulator
        vtool_platform=iossim
        ;;
    device)
        target=aarch64-apple-ios
        platform=iPhoneOS
        vtool_platform=ios
        ;;
    *)
        echo "usage: ios/build.sh [sim|device]" >&2
        exit 2
        ;;
esac

rustup target add "$target" >/dev/null
cargo build --release --target "$target"

app="target/$target/release/Sandy 3.app"
rm -rf "$app"
mkdir -p "$app"
# UIKit from the iOS 27 SDK stops an app dead at launch if it has not
# adopted the UIScene lifecycle, which winit 0.30 has not, whereas one built
# against the 26 SDK is let through with a warning. The binary is the same
# either way; what UIKit looks at is the SDK version stamped on it, so that
# is set to 26 here. This can go once winit does scenes.
xcrun vtool -set-build-version "$vtool_platform" 16.0 26.0 -replace \
    -output "$app/sandy-3" "target/$target/release/sandy-3"
sed "s/__PLATFORM__/$platform/" ios/Info.plist > "$app/Info.plist"

case "$kind" in
    sim)
        # The simulator takes an ad hoc signature.
        codesign --force --sign - "$app"
        if ! xcrun simctl list devices booted | grep -q Booted; then
            xcrun simctl boot "iPhone 17"
        fi
        # Xcode 27 replaced the Simulator app with Device Hub, which shows
        # simulators and phones alike; an older Xcode still has the app.
        # Whichever is there is brought up. The install and launch below
        # need neither, and nor does `xcrun simctl io booted screenshot`.
        hub="$(xcode-select -p)/../Applications/DeviceHub.app"
        if [ -d "$hub" ]; then
            open "$hub"
        else
            open -b com.apple.iphonesimulator 2>/dev/null || true
        fi
        xcrun simctl install booted "$app"
        echo "Installed $app. Launching; the log follows."
        xcrun simctl launch --console booted com.playforge.sandy3
        ;;
    device)
        : "${IOS_SIGNING_IDENTITY:?set IOS_SIGNING_IDENTITY to a development signing identity}"
        : "${IOS_PROVISIONING_PROFILE:?set IOS_PROVISIONING_PROFILE to a .mobileprovision for com.playforge.sandy3}"
        cp "$IOS_PROVISIONING_PROFILE" "$app/embedded.mobileprovision"
        # The entitlements a development build needs are the ones in the
        # profile; pull them out and sign with them.
        entitlements=$(mktemp -t sandy-entitlements).plist
        security cms -D -i "$IOS_PROVISIONING_PROFILE" \
            | plutil -extract Entitlements xml1 -o "$entitlements" -
        codesign --force --sign "$IOS_SIGNING_IDENTITY" --entitlements "$entitlements" "$app"
        rm -f "$entitlements"
        echo "Built and signed $app."
        echo "Install it with: xcrun devicectl device install app --device <name or id> \"$app\""
        ;;
esac
