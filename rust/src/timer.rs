use std::time::SystemTime;

pub struct Timer {
    t: SystemTime,
    millis: u128
}

impl Timer {

    pub fn new () -> Self {
        Timer {
            t: SystemTime::now(),
            millis: 0,
        }
    }

    pub fn start (&mut self) {
        self.t = SystemTime::now();
    }

    pub fn stop (&mut self) -> u128 {
        match self.t.elapsed() {
            Ok(elapsed) => {
                self.millis = elapsed.as_millis();
                return self.millis;
            }
            Err(e) => {
                panic!("Error: {e:?}")
            }
        }
    }
}

