//Hàm

function LayPath(){
  const filePath = document.getElementById("fileImage").value;
  document.getElementById("test1").innerHTML = filePath + filePath.length;
}