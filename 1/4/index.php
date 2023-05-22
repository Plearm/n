<?php 
	header('Content-Type: text/html; charset=UTF-8');

	if ($_SERVER['REQUEST_METHOD'] == 'GET') {
    $messages = array();
    if (!empty($_COOKIE['save'])) {
      setcookie('save', '', 100000);
      $messages[] = 'Спасибо, результаты сохранены.';
    }
    $errors = array();
    $errors['uName'] = !empty($_COOKIE['uName_error']);
  	$errors['uMail'] = !empty($_COOKIE['uMail_error']);
  	$errors['uDate'] = !empty($_COOKIE['uDate_error']);
    $errors['uGen'] = !empty($_COOKIE['uGen_error']);
    $errors['uLim'] = !empty($_COOKIE['uLim_error']);
    $errors['uBio'] = !empty($_COOKIE['uBio_error']);

    if ($errors['uName']) {
      setcookie('uName_error', '', 100000);
      $messages[] = '<div class="error">Заполните имя.</div>';
    }
	  if ($errors['uMail']) {
      setcookie('uMail_error', '', 100000);
      $messages[] = '<div class="error">Заполните e-mail.</div>';
    }
	  if ($errors['uDate']) {
      setcookie('uDate_error', '', 100000);
      $messages[] = '<div class="error">Заполните дату рождения.</div>';
    }
    if ($errors['uGen']) {
      setcookie('uGen_error', '', 100000);
      $messages[] = '<div class="error">Заполните Пол.</div>';
    }
    if ($errors['uLim']) {
      setcookie('uLim_error', '', 100000);
      $messages[] = '<div class="error">Заполните конечности.</div>';
    }
	  if ($errors['uBio']) {
      setcookie('uBio_error', '', 100000);
      $messages[] = '<div class="error">Заполните биографию.</div>';
    }
    print($messages);

    $values = array();
    $values['uName'] = empty($_COOKIE['uName_value']) ? '' : $_COOKIE['uName_value'];
	  $values['uMail'] = empty($_COOKIE['uMail_value']) ? '' : $_COOKIE['uMail_value'];
	  $values['uDate'] = empty($_COOKIE['uDate_value']) ? '' : $_COOKIE['uDate_value'];
    $values['uGen'] = empty($_COOKIE['uGen_value']) ? '' : $_COOKIE['uGen_value'];
    $values['uLim'] = empty($_COOKIE['uLim_value']) ? '' : $_COOKIE['uLim_value'];
	  $values['uBio'] = empty($_COOKIE['uBio_value']) ? '' : $_COOKIE['uBio_value'];

    include('form.php');
} else {
  
	$errors = false;
  if (empty($_POST['uName'])) {
    setcookie('uName_error', '1',time() + 30 * 24 * 60 * 60);
    $errors = TRUE;
  }
  if (empty($_POST['uMail'])) {
    setcookie('uMail_error', '1',time() + 30 * 24 * 60 * 60);
    $errors = TRUE;
  }
  if (empty($_POST['uDate'])) {
    setcookie('uDate_error', '1',time() + 30 * 24 * 60 * 60);
    $errors = TRUE;
  }
  if (empty($_POST['uGen'])) {
    setcookie('uGen_error', '1',time() + 30 * 24 * 60 * 60);
    $errors = TRUE;
  }
  if (empty($_POST['uLim'])) {
    setcookie('uLim_error', '1',time() + 30 * 24 * 60 * 60);
    $errors = TRUE;
  }
  if (empty($_POST['uBio'])) {
    setcookie('uBio_error', '1',time() + 30 * 24 * 60 * 60);
    $errors = TRUE;
  }
	if ($errors) {
    setcookie('uName_value', '', 100000);
		setcookie('uMail_value', '', 100000);
		setcookie('uDate_value', '', 100000);
    setcookie('uGen_value', '', 100000);
    setcookie('uLim_value', '', 100000);
		setcookie('uBio_value', '', 100000);
    header('Location: index.php');
    exit();
  }else {
    setcookie('uName_value', $_POST['uName'], time() + 30 * 24 * 60 * 60);
    setcookie('uMail_value', $_POST['uMail'], time() + 30 * 24 * 60 * 60);
    setcookie('uDate_value', $_POST['uDate'], time() + 30 * 24 * 60 * 60);
    setcookie('uGen_value', $_POST['uGen'], time() + 30 * 24 * 60 * 60);
    setcookie('uLim_value', $_POST['uLim'], time() + 30 * 24 * 60 * 60);
    setcookie('uBio_value', $_POST['uBio'], time() + 30 * 24 * 60 * 60);
	}
	try {
        $uName = $_POST['uName'];
        $uMail = $_POST['uMail'];
        $uDate = $_POST['uDate'];
        $uGen = $_POST['uGen'];
        $uLim = $_POST['uLim'];
        $uBio = $_POST['uBio'];
        $user = 'u47755';
        $pass = '2914865';
        $db = new PDO('mysql:host=localhost;dbname=u47755', $user, $pass, array(PDO::ATTR_PERSISTENT => true));
        $stmt = $db->prepare("INSERT INTO web4 (name, email, date, gen, lim, bio) VALUES (:name, :email, :date, :gen, :lim,  :bio)");
        $stmt->bindParam(':name', $uName);
        $stmt->bindParam(':email', $uMail);
        $stmt->bindParam(':date', $uDate);
        $stmt->bindParam(':gen', $uGen);
        $stmt->bindParam(':lim', $uLim);
        $stmt->bindParam(':bio', $uBio);
        if($stmt->execute()==false){
          print_r($stmt->errorCode());
          print_r($stmt->errorInfo());
          exit();
        }
    } catch (PDOException $e) {
        print('Error : ' . $e->getMessage());
        exit();
    }
	setcookie('save', '1');
  header('Location: index.php');
  print_r("Added");
}
?>
