<?php
header('Content-Type: text/html; charset=UTF-8');
if ($_SERVER['REQUEST_METHOD'] != 'POST'){
	print_r('Does not accept non POST methods');
}
$errors = FALSE;
if(empty($_POST['field-name-1']) || !isset($_POST['field-name-4']) || empty($_POST['field-email']) || empty($_POST['field-date']) || empty($_POST['bio-field']) || empty($_POST['checkbox']) || $_POST['checkbox'] == false){
	print_r('Empty fields!');
	exit();
}
if(!is_numeric($_POST['radio-group-2'])){
	print_r('Limb field is non-numeric');
}
print_r("Non null data... <br/>");
$name = $_POST['field-name-1'];
$email = $_POST['field-email'];
$birth = $_POST['field-date'];
$sex = $_POST['radio-group-1'];
$limbs = intval($_POST['radio-group-2']);
$superpowers = $_POST['field-name-4'];
$bio= $_POST['bio-field'];
$bioregex = "/^\s*\w+[\w\s\.,-]*$/";
$regex = "/^\w+[\w\s-]*$/";
$dateregex = "/^\d{4}-\d{2}-\d{2}$/";
$mailregex = "/^[\w\.-]+@([\w-]+\.)+[\w-]{2,4}$/";
$super_list = array('immortality','walkthroughwalls','levitation');
if(!preg_match($regex,$name)){
	print_r('Invalid name format');
	exit();
}
if($limbs < 1 || $limbs > 5){
	print_r('Invalid am of limbs');
	exit();
}
if(!preg_match($dateregex,$birth)){
	print_r('Invalid birth format');
	exit();
}
preg_match_all("/\d+/",$birth,$matches);
if (!checkdate($matches[0][1],$matches[0][2],$matches[0][0])){
	print_r('Date does not exist');
	exit();
}
if(!preg_match($bioregex,$bio)){
	print_r('Invalid bio format');
	exit();
}
if(!preg_match($mailregex,$email)){
	print_r('Invalid email format');
	exit();
}
if($sex !== 'male' && $sex !== 'female'){
	print_r('Invalid sex format');
	exit();
}
foreach($superpowers as $checking){
	if(array_search($checking,$super_list)=== false){
			print_r('Invalid superpower value!');
			exit();
	}
}
$user = 'u52830';
$pass = '7841698';
$db = new PDO('mysql:host=localhost;dbname=u52830', $user, $pass, array(PDO::ATTR_PERSISTENT => true));
try {
  $stmt = $db->prepare("INSERT INTO bentex SET name=:name, email=:email, birthdate=:birthdate, sex=:sex, limb_count=:limbs, bio=:bio");
  $stmt->bindParam(':name', $name);
  $stmt->bindParam(':email', $email);
  $stmt->bindParam(':birthdate', $birth);
  $stmt->bindParam(':sex', $sex);
  $stmt->bindParam(':limbs', $limbs);
  $stmt->bindParam(':bio', $bio);
  if($stmt->execute()==false){
  print_r($stmt->errorCode());
  print_r($stmt->errorInfo());
  exit();
  }
  $id = $db->lastInsertId();
  $sppe= $db->prepare("INSERT INTO superpowers SET name=:name, person_id=:person");
  $sppe->bindParam(':person', $id);
  foreach($superpowers as $inserting){
	$sppe->bindParam(':name', $inserting);
	if($sppe->execute()==false){
	  print_r($sppe->errorCode());
	  print_r($sppe->errorInfo());
	  exit();
	}
  }
} 
catch(PDOException $e){
  print('Error : ' . $e->getMessage());
  exit();
}

print_r("Succesfully added new stuff, probably...");
?>
