<?php
include('lib.php');
header('Content-Type: text/html; charset=UTF-8');
$user = 'u47755';
$pass = '2914865';
function validateArray($arr){
	foreach($arr as $checking){
		if ($checking === 1) return false;
	}
	return true;
}
if ($_SERVER['REQUEST_METHOD'] == 'GET'){
	  // Массив для временного хранения сообщений пользователю.
	  $messages = array();
	  $messages['saved']='';
	  $messages['passmessage'] = '';
	  if (!empty($_COOKIE['save'])) {
		setcookie('save', '', 100000);
		setcookie('login', '', 100000);
		setcookie('pass', '', 100000);
		$messages['saved'] = 'Thank you, your results were saved';
		if (!empty($_COOKIE['pass'])) {
			$messages['passmessage'] = sprintf('You can <a href="login.php">log in</a> with login <strong>%s</strong>
			  and password <strong>%s</strong> для изменения данных.',
			  strip_tags($_COOKIE['login']),
			  strip_tags($_COOKIE['pass']));
		  }
	  }
	
	  $errors = array();
	  $errors['name'] = !empty($_COOKIE['name_error']);
	  $errors['email'] = !empty($_COOKIE['email_error']);
	  $errors['birth_date'] = !empty($_COOKIE['birth_error']);
	  $errors['sex'] = !empty($_COOKIE['sex_error']);
	  $errors['limbs'] = !empty($_COOKIE['limb_error']);
	  $errors['super'] = !empty($_COOKIE['super_error']);
	  $errors['bio'] = !empty($_COOKIE['bio_error']);
	  $errors['check'] = !empty($_COOKIE['check_error']);

	  if ($errors['name']) {
		setcookie('name_error', '', 100000);
		$messages['bad_name'] = '<span class="error-text">Name must contain at least 1 letter, can only have letters, spacing and -</span>';
	  }
	  if ($errors['email']) {
		setcookie('email_error', '', 100000);
		$messages['bad_email'] = '<span class="error-text">Email can only contain letters, dots, dashes, @ sign, and email domain can have 2-4 letters.</span>';
	  }
	  if ($errors['birth_date']) {
		setcookie('birth_error', '', 100000);
		$messages['bad_date'] = '<span class="error-text">Must be filled.</span>';
	  }
	  if ($errors['sex']) {
		setcookie('sex_error', '', 100000);
		$messages['bad_sex'] = '<span class="error-text">Invalid choice.</span>';
	  }
	  if ($errors['limbs']) {
		setcookie('limb_error', '', 100000);
		$messages['bad_limbs'] = '<span class="error-text">Invalid choice.</span>';
	  }
	  if ($errors['super']) {
		setcookie('super_error', '', 100000);
		$messages['bad_super'] = '<span class="error-text">Must pick at least 1.</span>';
	  }
	  if ($errors['bio']) {
		setcookie('bio_error', '', 100000);
		$messages['bad_bio'] = '<span class="error-text">Bio must contain at least 1 letter, can only have letters, spacing and -</span>';
	  }
	  if ($errors['check']) {
		setcookie('check_error', '', 100000);
		$messages['bad_check'] = '<span class="error-text">Must be signed!</span>';
	  }
	
	  $values = array();
	  $values['name'] = empty($_COOKIE['name_value']) ? '' : strip_tags($_COOKIE['name_value']);
	  $values['email'] = empty($_COOKIE['email_value']) ? '' : strip_tags($_COOKIE['email_value']);
	  $values['birth_date'] = empty($_COOKIE['birth_value']) ? '' : strip_tags($_COOKIE['birth_value']);
	  $values['sex'] = empty($_COOKIE['sex_value']) ? '' : strip_tags($_COOKIE['sex_value']);
	  $values['limbs'] = empty($_COOKIE['limb_value']) ? '' : intval($_COOKIE['limb_value']);
	  $values['super'] = empty($_COOKIE['super_value']) ? '' : json_decode($_COOKIE['super_value']);
	  $values['bio'] = empty($_COOKIE['bio_value']) ? '' : strip_tags($_COOKIE['bio_value']);
	  //print_r(empty($errors));
	  //print_r(!empty($_COOKIE[session_name()]));
	  //print_r(session_start());
	  //print_r(!empty($_SESSION['login']));

	  if (validateArray($errors) && !empty($_COOKIE[session_name()]) &&
      session_start() && !empty($_SESSION['login'])){
		$db = connectToDB($user,$pass);
		try {
			$id = $_SESSION['uid'];
			$pdostate = $db->prepare("SELECT name,email,birthdate,sex,limb_count,bio FROM contracts WHERE id=:id");
			$superstate = $db->prepare("SELECT name FROM superpowers WHERE person_id=:id");
			$pdostate->bindParam(':id',$id);
			$superstate->bindParam(':id',$id);
			if($pdostate->execute()==false) {
				print_r($pdostate->errorCode());
				print_r($pdostate->errorInfo());
				exit();
			}
			if($superstate->execute()==false) {
				print_r($superstate->errorCode());
				print_r($superstate->errorInfo());
				exit();
			}
			$dbread = $pdostate->fetch(PDO::FETCH_ASSOC);
			$dbread['super'] = $superstate->fetchAll(PDO::FETCH_COLUMN,0);
			$values['name'] = strip_tags($dbread['name']);
			$values['email'] = strip_tags($dbread['email']);
			$values['birth_date'] = strip_tags($dbread['birthdate']);
			$values['sex'] = strip_tags($dbread['sex']);
			$values['limbs'] = intval($dbread['limb_count']);
			$values['super'] = $dbread['super'];
			$values['bio'] = strip_tags($dbread['bio']);

		}catch(PDOException $e){
			print('Error : ' . $e->getMessage());
			exit();
		}
		printf('Вход с логином %s, uid %d', $_SESSION['login'], $_SESSION['uid']);
		}
	include('form.php');
	 }
else {
$errors = FALSE;
$regex = "/^\s*\w+[\w\s-]*$/";
$bioregex = "/^\s*\w+[\w\s\.,!\?-]*$/";
$dateregex = "/^\d{4}-\d{2}-\d{2}$/";
$mailregex = "/^[\w\.-]+@([\w-]+\.)+[\w-]{2,4}$/";
$super_list = array('immortality','walkthroughwalls','levitation');

if(empty($_POST['field-name-1']) || !preg_match($regex,$_POST['field-name-1'])){
    setcookie('name_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('name_value', $_POST['field-name-1'], time() + 30 * 24 * 60 * 60);
}

if (empty($_POST['field-email']) || !preg_match($mailregex,$_POST['field-email'])){
    setcookie('email_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('email_value', $_POST['field-email'], time() + 30 * 24 * 60 * 60);
}

$date_correct = !empty($_POST['field-date']);
if($date_correct){
	$date_correct=preg_match($dateregex,$_POST['field-date']);
	if($date_correct){
		preg_match_all("/\d+/",$_POST['field-date'],$matches);
		$date_correct=checkdate($matches[0][1],$matches[0][2],$matches[0][0]);
	}
}

if (!$date_correct){
    setcookie('birth_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('birth_value', $_POST['field-date'], time() + 30 * 24 * 60 * 60);
}

if (empty($_POST['radio-group-1']) || ($_POST['radio-group-1']!=='male' && $_POST['radio-group-1']!=='female')){
    setcookie('sex_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('sex_value', $_POST['radio-group-1'], time() + 30 * 24 * 60 * 60);
}

if ((!is_numeric($_POST['radio-group-2'])) || (intval($_POST['radio-group-2']) < 1) || (intval($_POST['radio-group-2']) > 5)){
    setcookie('limb_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('limb_value', $_POST['radio-group-2'], time() + 30 * 24 * 60 * 60);
}

$super_correct = !empty($_POST['field-name-4']);
if($super_correct) {
	foreach($_POST['field-name-4'] as $checking){
		if(array_search($checking,$super_list)=== false){
			$super_correct = FALSE;
			break;
		}
	}
}
if (!$super_correct){
    setcookie('super_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('super_value', json_encode($_POST['field-name-4']), time() + 30 * 24 * 60 * 60);
}

if(empty($_POST['bio-field']) || !preg_match($bioregex,$_POST['bio-field'])){
    setcookie('bio_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('bio_value', $_POST['bio-field'], time() + 30 * 24 * 60 * 60);
}
if(empty($_POST['checkbox']) && $_POST['checkbox']!='realslim'){
	setcookie('check_error', '1', time() + 24 * 60 * 60);
    $errors = TRUE;
}
else {
	setcookie('check_value', $_POST['checkbox'], time() + 30 * 24 * 60 * 60);
}
if ($errors) {
    header('Location: index.php');
    exit();
}
else {
	setcookie('name_error', '', 100000);
	setcookie('email_error', '', 100000);
	setcookie('birth_error', '', 100000);
	setcookie('sex_error', '', 100000);
	setcookie('limb_error', '', 100000);
	setcookie('super_error', '', 100000);
	setcookie('bio_error', '', 100000);
}
$name = $_POST['field-name-1'];
$email = $_POST['field-email'];
$birth = $_POST['field-date'];
$sex = $_POST['radio-group-1'];
$limbs = intval($_POST['radio-group-2']);
$superpowers = $_POST['field-name-4'];
$bio= $_POST['bio-field'];
$db = connectToDB($user,$pass);
if (!empty($_COOKIE[session_name()]) &&
session_start() && !empty($_SESSION['login']) && !empty($_SESSION['uid'])) {
// TODO: перезаписать данные в БД новыми данными,
// кроме логина и пароля.
	try{
		$stmt = $db->prepare("UPDATE contracts SET name=:name, email=:email, birthdate=:birthdate, sex=:sex, limb_count=:limbs, bio=:bio WHERE id=:this_id");
		$stmt->bindParam(':name', $name);
		$stmt->bindParam(':email', $email);
		$stmt->bindParam(':birthdate', $birth);
		$stmt->bindParam(':sex', $sex);
		$stmt->bindParam(':limbs', $limbs);
		$stmt->bindParam(':bio', $bio);
		$stmt->bindParam(':this_id', $_SESSION['uid']);
		if($stmt->execute()==false){
			print_r($stmt->errorCode());
			print_r($stmt->errorInfo());
			exit();
		}
		$spped = $db->prepare("DELETE FROM superpowers WHERE person_id=:this_id");
		$spped->bindParam(':this_id', $_SESSION['uid']);
		if($spped->execute()==false){
			print_r($sppe->errorCode());
			print_r($sppe->errorInfo());
			exit();
		}
		$sppe= $db->prepare("INSERT INTO superpowers SET name=:name, person_id=:person");
		$sppe->bindParam(':person', $_SESSION['uid']);
		foreach($superpowers as $inserting){
			$sppe->bindParam(':name', $inserting);
			if($sppe->execute()==false){
			print_r($sppe->errorCode());
			print_r($sppe->errorInfo());
			exit();
			}
		}
	} catch(PDOException $e){
		print('Error : ' . $e->getMessage());
		exit();
		}
}
else {
	try {
	$stmt = $db->prepare("INSERT INTO contracts SET name=:name, email=:email, birthdate=:birthdate, sex=:sex, limb_count=:limbs, bio=:bio");
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
	$loginn = uniqid('u',true);
	$passok = uniqid('',true).strval(rand(0,100));
	$pass_user = password_hash($passok, PASSWORD_DEFAULT);
	$logpdostate = $db->prepare("INSERT INTO login SET p_id=:id, login=:login, pass_hash=:hash");
	$logpdostate->bindParam(':id',$id);
	$logpdostate->bindParam(':login',$loginn);
	$logpdostate->bindParam(':hash',$pass_user);
	if($logpdostate->execute() == false){
		print_r($logpdostate->errorCode());
		print_r($logpdostate->errorInfo());
		exit();
	}
	setcookie('login', $loginn);
	setcookie('pass', $passok);
	} 


	catch(PDOException $e){
	print('Error : ' . $e->getMessage());
	exit();
	}
}
print_r("Succesfully added new stuff, probably...");
setcookie('save', '1');
header('Location: index.php');
}
?>
