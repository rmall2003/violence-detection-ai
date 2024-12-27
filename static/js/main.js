let map;
        let geocoder;
     
        function initMap() {
            geocoder = new google.maps.Geocoder();
        }

        function getLocation() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(showPosition, showError);
            } else {
                alert("Geolocation is not supported by this browser.");
            }
        }

        function showPosition(position) {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            console.log("Latitude: " + lat + ", Longitude: " + lon);

            const latlng = { lat: parseFloat(lat), lng: parseFloat(lon) };
            
            geocoder.geocode({ location: latlng }, (results, status) => {
                if (status === "OK") {
                    if (results[0]) {
                        document.getElementById('location').value = results[0].formatted_address;
                        console.log(results[0].formatted_address);
                    } else {
                        alert('No results found');
                    }
                } else {
                    alert('Geocoder failed due to: ' + status);
                }
            });
        }

        function showError(error) {
    let errorMessage = "";
    switch (error.code) {
        case error.PERMISSION_DENIED:
            errorMessage = "Location permission denied. Please enable location access in your browser settings.";
            console.log(error);
            document.getElementById('location').value = "Location permission denied";
            showEnableLocationInstructions();
            break;
        case error.POSITION_UNAVAILABLE:
            errorMessage = "Location information is unavailable.";
            document.getElementById('location').value = "Location unavailable";
            break;
        case error.TIMEOUT:
            errorMessage = "The request to get user location timed out.";
            document.getElementById('location').value = "Request timed out";
            break;
        case error.UNKNOWN_ERROR:
            errorMessage = "An unknown error occurred.";
            document.getElementById('location').value = "Unknown error";
            break;
    }
    alert(errorMessage);
}

function showEnableLocationInstructions() {
    const instructions = `
        <div id="location-instructions" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-white rounded-lg p-6 w-11/12 max-w-md text-center">
                <h2 class="text-2xl font-bold mb-4">Enable Location Services</h2>
                <p class="mb-6">
                    To use this feature, please enable location services in your browser settings:
                    <ul class="list-disc list-inside text-left mt-2">
                        <li>Click on the padlock icon in the address bar.</li>
                        <li>Select "Site settings".</li>
                        <li>Change the Location permission to "Allow".</li>
                        <li>Reload this page and click "Get My Location" again.</li>
                    </ul>
                </p>
                <button onclick="closeInstructions()" 
                    class="bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 transition">
                    Got It!
                </button>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', instructions);
}

function closeInstructions() {
    const modal = document.getElementById('location-instructions');
    if (modal) modal.remove();
}


function goFullscreen() {
    const element = document.documentElement;

    if (element.requestFullscreen) {
        element.requestFullscreen();
    } else if (element.mozRequestFullScreen) { 
        element.mozRequestFullScreen();
    } else if (element.webkitRequestFullscreen) { 
        element.webkitRequestFullscreen();
    } else if (element.msRequestFullscreen) { 
        element.msRequestFullscreen();
    }
}