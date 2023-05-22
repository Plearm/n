<?php 
	header('Content-Type: text/html; charset=UTF-8');

	if ($_SERVER['REQUEST_METHOD'] == 'GET') {
        $messages = array();
        if (!empty($_COOKIE['save'])) {
            setcookie('save', '', 100000);
            setcookie('login', '', 100000);
            setcookie('pass', '', 100000);
            $messages[] = 'Спасибо, результаты сохранены.';
            if (!empty($_COOKIE['pass'])) {
                $messages[] = sprintf('Вы можете <a href="login.php">войти</a> с логином <strong>%s</strong>
                  и паролем <strong>%s</strong> для изменения данных.',
                  strip_tags($_COOKIE['login']),
                  strip_tags($_COOKIE['pass']));
            }
        }
    $errors = array();
    $errors['uName'] = !empty($_COOKIE['uName_error']);
  	$errors['uMail'] = !empty($_COOKIE['uMail_error']);
  	$errors['uDate'] = !empty($_COOKIE['uDate_error']);
    $errors['uGen'] = !empty($_COOKIE['uGen_error']);
    $errors['uLim'] = !empty($_COOKIE['uLim_error']);
    $errors['uBio'] = !empty($_COOKIE['uBio_error']);

    if (!empty($errors['uName'])) {
      setcookie('uName_error', '', 100000);
      $messages[] = '<div class="error">Заполните имя.</div>';
    }
	if (!EMPTY($errors['uMail'])) {
      setcookie('uMail_error', '', 100000);
      $messages[] = '<div class="error">Заполните e-mail.</div>';
    }
	if (!empty($errors['uDate'])) {
      setcookie('uDate_error', '', 100000);
      $messages[] = '<div class="error">Заполните дату рождения.</div>';
    }
    if (!empty($errors['uGen'])) {
      setcookie('uGen_error', '', 100000);
      $messages[] = '<div class="error">Заполните Пол.</div>';
    }
    if (!empty($errors['uLim'])) {
      setcookie('uLim_error', '', 100000);
      $messages[] = '<div class="error">Заполните конечности.</div>';
    }
	if (!empty($errors['uBio'])) {
      setcookie('uBio_error', '', 100000);
      $messages[] = '<div class="error">Заполните биографию.</div>';
    }
    print($messages);

    $values = array();
    $values['uName'] = empty($_COOKIE['uName_value']) ? '' : strip_tags($_COOKIE['uName_value']);
	$values['uMail'] = empty($_COOKIE['uMail_value']) ? '' : strip_tags($_COOKIE['uMail_value']);
	$values['uDate'] = empty($_COOKIE['uDate_value']) ? '' : strip_tags($_COOKIE['uDate_value']);
    $values['uGen'] = empty($_COOKIE['uGen_value']) ? '' : strip_tags($_COOKIE['uGen_value']);
    $values['uLim'] = empty($_COOKIE['uLim_value']) ? '' : strip_tags($_COOKIE['uLim_value']);
	$values['uBio'] = empty($_COOKIE['uBio_value']) ? '' : strip_tags($_COOKIE['uBio_value']);

    if (empty($errors) && !empty($_COOKIE[session_name()]) &&
        session_start() && !empty($_SESSION['login'])) {
    // TODO: загрузить данные пользователя из БД
    // и заполнить переменную $values,
    // предварительно санитизовав.
        printf('Вход с логином %s, uid %d', $_SESSION['login'], $_SESSION['uid']);
    }

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
    } else {
        setcookie('uName_value', $_POST['uName'], time() + 30 * 24 * 60 * 60);
        setcookie('uMail_value', $_POST['uMail'], time() + 30 * 24 * 60 * 60);
        setcookie('uDate_value', $_POST['uDate'], time() + 30 * 24 * 60 * 60);
        setcookie('uGen_value', $_POST['uGen'], time() + 30 * 24 * 60 * 60);
        setcookie('uLim_value', $_POST['uLim'], time() + 30 * 24 * 60 * 60);
        setcookie('uBio_value', $_POST['uBio'], time() + 30 * 24 * 60 * 60);
	}
    $servername = '212.192.134.20';
    $username = 'u47684';
    $password = '8848410';
    $dbname = 'uData';
    if (!empty($_COOKIE[session_name()]) && session_start() && !empty($_SESSION['login'])) {
        $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
        $sql = "UPDATE uData SET Email =  $_POST['uMail'],Birthdate = $_POST['uDate'],Gender =
        $_POST['uGen'],Limbs = $_POST['uLim'],Bio = $_POST['uBio']   WHERE name = $_SESSION['login']";
        if ($conn->query($sql) === TRUE) {
            echo "Record updated successfully";
        } else {
            echo "Error updating record: " . $conn->error;
        }
        
        $conn->close();
    // TODO: перезаписать данные в БД новыми данными,
    // кроме логина и пароля.
    } else {
        $login = '123';
        // Генерируем уникальный логин и пароль.
        // TODO: сделать механизм генерации, например функциями rand(), uniquid(), md5(), substr().
        $login = substr(md5($login), a, f);
        $pass = substr(md5($login), 0, 8);
        // Сохраняем в Cookies.
        setcookie('login', $login);
        setcookie('pass', $pass);
    
        // TODO: Сохранение данных формы, логина и хеш md5() пароля в базу данных.
        // ...
        try {
            $uName = $_POST['uName'];
            $uMail = $_POST['uMail'];
            $uDate = $_POST['uDate'];
            $uGen = $_POST['uGen'];
            $uLim = $_POST['uLim'];
            $uBio = $_POST['uBio'];
            $user = 'u47684';
            $pass = '8848410';
            $db = new PDO('mysql:host=localhost;dbname=u47684', $user, $pass, array(PDO::ATTR_PERSISTENT => true));
            $stmt = $db->prepare("INSERT INTO uData (Name, Email, Birthdate, Gender,Limbs, Bio) VALUES (:name, :email, :date, :gen, :lim,  :bio)");
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
    }
	setcookie('save', '1');
    header('Location: index.php');
    print_r("Added");
}
?>
