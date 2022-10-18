var fileName;
var folderName = "C:\\Users\\tiem.nv164039\\PycharmProjects\\XLA\\web_base\\assets\\images\\";

$(document).ready(function () {
  $("#img1").change(function (e) {
    fileName = folderName + e.target.files[0].name;
  });
});

function xulyanh() {
  // get type xy ly anh
  let check = [];
  if ($("#check-11")[0].checked) check.push("CH");
  if ($("#check-12")[0].checked) check.push("CM");
  if ($("#check-21")[0].checked) check.push("LBP");
  if ($("#check-22")[0].checked) check.push("HOG");
  // console.log(check);

  // get list files name
  let files = $("#img2").prop("files");
  let files_names = $.map(files, function (val) {
    return folderName + val.name;
  });
  // console.log(files_names);

  // get thuoc do
  let thuocdo = $("#duoi").val();
  // console.log(thuocdo)

  // console.log(fileName);

  // gui du lieu toi server de xu ly
  $.ajax({
    url: "/xulyanh/",
    dataType: "json",
    type: "GET",
    data: {
      fileName: fileName,
      listFiles: files_names,
      listMethod: check,
      ruler: thuocdo,
    },
    success: function (data) {
//    debugger

      $('.anh1').css("background-image", `url(${data.data[0]})`);
      $('.anh2').css("background-image", `url(${data.data[1]})`);
      $('.anh3').css("background-image", `url(${data.data[2]})`);
       $('.anhgoc').css("background-image", `url(${data.data[3]})`);
    },
    error: function () {
      console.log("Loi roi");
    },
  });
}
