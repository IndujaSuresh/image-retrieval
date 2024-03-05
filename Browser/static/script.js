// JavaScript code goes here
// Selecting all required elements
const dropArea = document.querySelector(".drag-area"),
  dragText = dropArea.querySelector("header"),
  button = dropArea.querySelector("button"),
  input = dropArea.querySelector("input");
let file; // This is a global variable and we'll use it inside multiple functions

// Function to initialize event listeners
function initializeEventListeners() {
  // Click event listener for the button
  button.onclick = () => {
    input.click(); // If user clicks on the button then the input also clicked
  }

  // Change event listener for the input
  input.addEventListener("change", function () {
    // Getting user selected file and [0] means if user selects multiple files then we'll select only the first one
    file = this.files[0];
    dropArea.classList.add("active");
    showFile(); // Calling function to display the selected file
  });

  // Dragover event listener for the drop area
  dropArea.addEventListener("dragover", (event) => {
    event.preventDefault(); // Preventing default behavior
    dropArea.classList.add("active");
    dragText.textContent = "Release to Upload File";
  });

  // Dragleave event listener for the drop area
  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("active");
    dragText.textContent = "Drag & Drop to Upload File";
  });

  // Drop event listener for the drop area
  dropArea.addEventListener("drop", (event) => {
    event.preventDefault(); // Preventing default behavior
    file = event.dataTransfer.files[0]; // Getting user selected file
    showFile(); // Calling function to display the selected file
  });
}

// Calling the function to initialize event listeners
initializeEventListeners();
// Function to display the selected file
// Function to display the selected file
function showFile() {
  let fileType = file.type; // Getting selected file type
  let validExtensions = ["image/jpeg", "image/jpg", "image/png"]; // Array of valid image extensions
  if (validExtensions.includes(fileType)) { // If the selected file is an image file
    let fileReader = new FileReader(); // Creating new FileReader object
    fileReader.onload = () => {
      let fileURL = fileReader.result; // Passing user file source to fileURL variable
      let img = new Image();
      img.onload = function() {
        let aspectRatio = img.width / img.height;
        let dragAreaAspectRatio = 500 / 300; // Aspect ratio of the drag area
        let width, height;

        if (aspectRatio > dragAreaAspectRatio) {
          width = 500; // Set width to the width of the drag area
          height = width / aspectRatio;
        } else {
          height = 300; // Set height to the height of the drag area
          width = height * aspectRatio;
        }

        let imgTag = `<img id="uploadedImage" src="${fileURL}" alt="" style="width: ${width}px; height: ${height}px;">`; // Creating img tag with the selected file source and adjusted size
        dropArea.innerHTML = imgTag; // Adding the img tag inside dropArea container
      };
      img.src = fileURL;
    }
    fileReader.readAsDataURL(file);
    
  } else {
    alert("This is not an Image File!");
    dropArea.classList.remove("active");
    dragText.textContent = "Drag & Drop to Upload File";
  }
}



// Event listener for the Clear button
document.getElementById("clearButton").addEventListener("click", function(event) {
  // Prevent the default form submission behavior
  event.preventDefault();
  
  // Reload the current page
  window.location.reload();
});




