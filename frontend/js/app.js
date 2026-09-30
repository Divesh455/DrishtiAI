// =====================================================
// DrishtiAI Frontend
// =====================================================


// =====================================================
// CONFIGURATION
// =====================================================

// IMPORTANT:
//
// When opening frontend on your LAPTOP:
//
// const API_BASE_URL =
//     "http://localhost:8000";
//
// When opening frontend on your PHONE:
//
// use your LAPTOP IPv4 address.
//
// Example:
//
// const API_BASE_URL =
//     "http://10.162.237.115:8000";

const API_BASE_URL =
    "http://10.162.237.115:8000";


// =====================================================
// GLOBAL VARIABLES
// =====================================================

let cameraStream = null;

let selectedImage = null;


// =====================================================
// GET ELEMENTS
// =====================================================

const homeScreen =
    document.getElementById(
        "homeScreen"
    );


const cameraScreen =
    document.getElementById(
        "cameraScreen"
    );


const previewScreen =
    document.getElementById(
        "previewScreen"
    );


const analysisScreen =
    document.getElementById(
        "analysisScreen"
    );


const resultScreen =
    document.getElementById(
        "resultScreen"
    );


const camera =
    document.getElementById(
        "camera"
    );


const canvas =
    document.getElementById(
        "captureCanvas"
    );


const cameraStatus =
    document.getElementById(
        "cameraStatus"
    );


const mobileCameraInput =
    document.getElementById(
        "mobileCameraInput"
    );


const previewImage =
    document.getElementById(
        "previewImage"
    );


// =====================================================
// SCREEN MANAGEMENT
// =====================================================

function showScreen(screen) {

    document
        .querySelectorAll(".screen")
        .forEach(
            item => {

                item.classList.remove(
                    "active"
                );

            }
        );


    screen.classList.add(
        "active"
    );

}


// =====================================================
// START SCAN
// =====================================================

async function startCamera() {

    showScreen(
        cameraScreen
    );


    cameraStatus.innerText =
        "Checking camera access...";


    // -------------------------------------------------
    // Check browser camera support
    // -------------------------------------------------

    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        cameraStatus.innerText =
            "Live camera is unavailable. " +
            "Use the Mobile Camera button below.";

        return;

    }


    // -------------------------------------------------
    // Try live camera
    // -------------------------------------------------

    try {

        cameraStatus.innerText =
            "Requesting camera permission...";


        cameraStream =
            await navigator.mediaDevices
                .getUserMedia({

                    video: {

                        facingMode: {
                            ideal: "environment"
                        },

                        width: {
                            ideal: 1280
                        },

                        height: {
                            ideal: 720
                        }

                    },

                    audio: false

                });


        camera.srcObject =
            cameraStream;


        cameraStatus.innerText =
            "Camera ready • Hold steady";


    }

    catch (error) {

        console.error(
            "Live camera unavailable:",
            error
        );


        cameraStatus.innerText =
            "Live camera unavailable. " +
            "Use the Mobile Camera button below.";

    }

}


// =====================================================
// STOP CAMERA
// =====================================================

function stopCamera() {

    if (
        !cameraStream
    ) {

        return;

    }


    cameraStream
        .getTracks()
        .forEach(
            track => {

                track.stop();

            }
        );


    cameraStream = null;


    camera.srcObject = null;

}


// =====================================================
// CAPTURE LIVE CAMERA IMAGE
// =====================================================

function captureImage() {

    if (
        !cameraStream ||
        !camera.videoWidth
    ) {

        showError(
            "Live camera is not available. " +
            "Please use the Mobile Camera button."
        );

        return;

    }


    canvas.width =
        camera.videoWidth;


    canvas.height =
        camera.videoHeight;


    const context =
        canvas.getContext(
            "2d"
        );


    context.drawImage(
        camera,
        0,
        0,
        canvas.width,
        canvas.height
    );


    canvas.toBlob(

        function(blob) {

            if (!blob) {

                showError(
                    "Could not capture image."
                );

                return;

            }


            stopCamera();


            selectedImage =
                blob;


            showPreview(
                blob
            );

        },

        "image/jpeg",

        0.90

    );

}


// =====================================================
// SHOW IMAGE PREVIEW
// =====================================================

function showPreview(
    imageFile
) {

    const imageUrl =
        URL.createObjectURL(
            imageFile
        );


    previewImage.src =
        imageUrl;


    showScreen(
        previewScreen
    );

}


// =====================================================
// MOBILE NATIVE CAMERA
// =====================================================

function handleMobileImage(
    event
) {

    const files =
        event.target.files;


    if (
        !files ||
        files.length === 0
    ) {

        return;

    }


    const file =
        files[0];


    console.log(
        "Mobile camera image:",
        file.name
    );


    // -------------------------------------------------
    // Validate image
    // -------------------------------------------------

    if (
        !file.type.startsWith(
            "image/"
        )
    ) {

        showError(
            "Please capture a valid image."
        );

        return;

    }


    selectedImage =
        file;


    showPreview(
        file
    );


    // Allow same image to be
    // selected again later.

    event.target.value = "";

}


// =====================================================
// SEND IMAGE TO BACKEND
// =====================================================

async function sendImageToBackend(
    imageFile
) {

    showScreen(
        analysisScreen
    );


    const analysisMessage =
        document.getElementById(
            "analysisMessage"
        );


    analysisMessage.innerText =
        "Uploading image...";


    const formData =
        new FormData();


    formData.append(
        "file",
        imageFile,
        "eye_capture.jpg"
    );


    try {

        // ---------------------------------------------
        // Quality
        // ---------------------------------------------

        analysisMessage.innerText =
            "Checking image quality...";


        const response =
            await fetch(

                API_BASE_URL +
                "/api/v1/screening/screen",

                {

                    method: "POST",

                    body: formData

                }

            );


        // ---------------------------------------------
        // HTTP error
        // ---------------------------------------------

        if (
            !response.ok
        ) {

            const errorText =
                await response.text();


            throw new Error(
                errorText ||
                "Screening request failed."
            );

        }


        // ---------------------------------------------
        // Classification
        // ---------------------------------------------

        analysisMessage.innerText =
            "Running DR classification...";


        const result =
            await response.json();


        // ---------------------------------------------
        // Explanation
        // ---------------------------------------------

        analysisMessage.innerText =
            "Generating AI explanation...";


        await sleep(
            300
        );


        displayResult(
            result
        );

    }

    catch (error) {

        console.error(
            "Backend error:",
            error
        );


        showScreen(
            previewScreen
        );


        showError(

            "Unable to connect to " +
            "DrishtiAI backend.\n\n" +

            "Make sure FastAPI is running " +
            "on port 8000 and that the " +
            "API IP address in app.js is correct."

        );

    }

}


// =====================================================
// DISPLAY RESULT
// =====================================================

function displayResult(
    result
) {

    showScreen(
        resultScreen
    );


    // -------------------------------------------------
    // Screening ID
    // -------------------------------------------------

    document.getElementById(
        "screeningId"
    ).innerText =
        result.screening_id ||
        "";


    // -------------------------------------------------
    // Rejected
    // -------------------------------------------------

    if (
        result.status === "rejected"
    ) {

        showError(

            result.message ||
            "Image quality is insufficient."

        );

        return;

    }


    // -------------------------------------------------
    // Classification
    // -------------------------------------------------

    const classification =
        result.classification ||
        {};


    document.getElementById(
        "gradeNumber"
    ).innerText =
        classification.grade ??
        "-";


    document.getElementById(
        "gradeLabel"
    ).innerText =
        classification.label ||
        "-";


    document.getElementById(
        "confidence"
    ).innerText =

        "Confidence: " +

        formatPercent(
            classification.confidence
        );


    // -------------------------------------------------
    // Quality
    // -------------------------------------------------

    const quality =
        result.quality ||
        {};


    document.getElementById(
        "quality"
    ).innerText =
        quality.status ||
        "-";


    // -------------------------------------------------
    // Lesions
    // -------------------------------------------------

    const lesions =
        result.lesions ||
        {};


    let lesionCount =
        0;


    if (
        lesions.lesion_count !==
        undefined
    ) {

        lesionCount =
            lesions.lesion_count;

    }
    else if (
        Array.isArray(
            lesions.lesions
        )
    ) {

        lesionCount =
            lesions.lesions.length;

    }


    document.getElementById(
        "lesions"
    ).innerText =
        lesionCount;


    displayLesionDetails(
        lesions
    );


    // -------------------------------------------------
    // Explanation
    // -------------------------------------------------

    const explanation =
        result.explainability ||
        {};


    document.getElementById(
        "explanationText"
    ).innerText =

        explanation.explanation ||

        "Grad-CAM++ explanation generated.";


    displayOverlay(
        explanation.overlay_path
    );

}


// =====================================================
// DISPLAY GRAD-CAM OVERLAY
// =====================================================

function displayOverlay(
    path
) {

    const image =
        document.getElementById(
            "overlayImage"
        );


    if (!path) {

        image.style.display =
            "none";

        return;

    }


    let cleanPath =
        path.replaceAll(
            "\\",
            "/"
        );


    const marker =
        cleanPath.indexOf(
            "backend/outputs/"
        );


    if (
        marker === -1
    ) {

        image.style.display =
            "none";

        return;

    }


    cleanPath =
        cleanPath.substring(

            marker +
            "backend/outputs/"
                .length

        );


    image.src =
        API_BASE_URL +
        "/outputs/" +
        cleanPath;


    image.style.display =
        "block";

}


// =====================================================
// LESION DETAILS
// =====================================================

function displayLesionDetails(
    lesions
) {

    const container =
        document.getElementById(
            "lesionDetails"
        );


    if (
        !Array.isArray(
            lesions.lesions
        ) ||
        lesions.lesions.length === 0
    ) {

        container.innerText =
            "No detected lesions.";

        return;

    }


    const counts =
        {};


    lesions.lesions.forEach(
        lesion => {

            const type =
                lesion.type ||
                "unknown";


            counts[type] =
                (
                    counts[type] ||
                    0
                ) + 1;

        }
    );


    container.innerHTML =
        "";


    Object.entries(
        counts
    ).forEach(
        ([type, count]) => {

            const row =
                document.createElement(
                    "div"
                );


            row.innerText =

                formatLesionName(
                    type
                ) +

                ": " +

                count;


            container.appendChild(
                row
            );

        }
    );

}


// =====================================================
// FORMAT LESION NAME
// =====================================================

function formatLesionName(
    name
) {

    return name

        .replaceAll(
            "_",
            " "
        )

        .replace(
            /\b\w/g,
            letter =>
                letter.toUpperCase()
        );

}


// =====================================================
// FORMAT PERCENTAGE
// =====================================================

function formatPercent(
    value
) {

    if (
        value === undefined ||
        value === null
    ) {

        return "-";

    }


    return (

        Number(value) * 100

    ).toFixed(1) + "%";

}


// =====================================================
// ERROR
// =====================================================

function showError(
    message
) {

    const box =
        document.getElementById(
            "errorBox"
        );


    const text =
        document.getElementById(
            "errorMessage"
        );


    text.innerText =
        message;


    box.classList.remove(
        "hidden"
    );

}


// =====================================================
// SLEEP
// =====================================================

function sleep(
    milliseconds
) {

    return new Promise(
        resolve =>

            setTimeout(
                resolve,
                milliseconds
            )

    );

}


// =====================================================
// EVENT: START SCAN
// =====================================================

document
    .getElementById(
        "startScanBtn"
    )
    .addEventListener(
        "click",
        startCamera
    );


// =====================================================
// EVENT: LIVE CAMERA CAPTURE
// =====================================================

document
    .getElementById(
        "captureBtn"
    )
    .addEventListener(
        "click",
        captureImage
    );


// =====================================================
// EVENT: MOBILE CAMERA
// =====================================================

document
    .getElementById(
        "mobileCameraInput"
    )
    .addEventListener(
        "change",
        handleMobileImage
    );


// =====================================================
// EVENT: CANCEL CAMERA
// =====================================================

document
    .getElementById(
        "cancelCameraBtn"
    )
    .addEventListener(
        "click",
        () => {

            stopCamera();

            showScreen(
                homeScreen
            );

        }
    );


// =====================================================
// EVENT: RETAKE
// =====================================================

document
    .getElementById(
        "retakeBtn"
    )
    .addEventListener(
        "click",
        () => {

            selectedImage =
                null;

            previewImage.src =
                "";

            startCamera();

        }
    );


// =====================================================
// EVENT: USE IMAGE
// =====================================================

document
    .getElementById(
        "useImageBtn"
    )
    .addEventListener(
        "click",
        () => {

            if (
                !selectedImage
            ) {

                showError(
                    "No image selected."
                );

                return;

            }


            sendImageToBackend(
                selectedImage
            );

        }
    );


// =====================================================
// EVENT: SCAN AGAIN
// =====================================================

document
    .getElementById(
        "scanAgainBtn"
    )
    .addEventListener(
        "click",
        startCamera
    );


// =====================================================
// EVENT: HOME
// =====================================================

document
    .getElementById(
        "homeBtn"
    )
    .addEventListener(
        "click",
        () => {

            stopCamera();

            showScreen(
                homeScreen
            );

        }
    );


// =====================================================
// EVENT: CLOSE ERROR
// =====================================================

document
    .getElementById(
        "errorCloseBtn"
    )
    .addEventListener(
        "click",
        () => {

            document
                .getElementById(
                    "errorBox"
                )
                .classList.add(
                    "hidden"
                );

        }
    );