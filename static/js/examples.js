document.addEventListener("DOMContentLoaded", function () {
  const images = [
    "./static/images/examples/example1.jpg",
    "./static/images/examples/example2.jpg",
    "./static/images/examples/example3.jpg",
    "./static/images/examples/example4.jpg",
    "./static/images/examples/example5.jpg",
    // add as many as needed
  ];

  let currentIndex = 0;
  const imgElement = document.getElementById("carouselImage");

  document.getElementById("prevExample").addEventListener("click", () => {
    currentIndex = (currentIndex - 1 + images.length) % images.length;
    imgElement.src = images[currentIndex];
  });

  document.getElementById("nextExample").addEventListener("click", () => {
    currentIndex = (currentIndex + 1) % images.length;
    imgElement.src = images[currentIndex];
  });
});
