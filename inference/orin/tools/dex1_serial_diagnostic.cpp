// Read-only-at-position Dex1-1 serial diagnostic.
//
// The DDS bridge must be stopped while this program owns the selected serial
// port.  It sends BRAKE-mode packets only, so it never requests a new joint
// position.  The returned mode, error, temperature and kinematics distinguish
// a live motor controller from a stale DDS state publisher.

#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: dex1_serial_diagnostic PORT MOTOR_ID\n";
        return 2;
    }
    const std::string port = argv[1];
    const int motor_id = std::stoi(argv[2]);
    if (motor_id < 0 || motor_id > 1) {
        std::cerr << "MOTOR_ID must be 0 (right) or 1 (left)\n";
        return 2;
    }

    SerialPort serial(port.c_str());
    MotorCmd command;
    command.motorType = MotorType::M4010;
    command.id = static_cast<unsigned short>(motor_id);
    command.mode = queryMotorMode(command.motorType, MotorMode::BRAKE);
    command.kp = 0.0F;
    command.kd = 0.0F;
    command.q = 0.0F;
    command.dq = 0.0F;
    command.tau = 0.0F;
    command.timeout = 1;

    MotorData state;
    state.motorType = MotorType::M4010;
    int successes = 0;
    for (int sample = 0; sample < 20; ++sample) {
        const bool ok = serial.sendRecv(&command, &state);
        std::cout << "sample=" << sample << " ok=" << ok;
        if (ok) {
            ++successes;
            const auto *raw = state.get_motor_recv_data();
            const auto winding_temp_c = static_cast<unsigned int>(raw[4]);
            const auto voltage_v = static_cast<double>(raw[5]) / 2.0;
            std::cout << " id=" << static_cast<int>(state.motor_id)
                      << " mode=" << static_cast<int>(state.mode)
                      << " error=" << state.merror
                      << " shell_temp_c=" << state.temp
                      << " winding_temp_c=" << winding_temp_c
                      << " voltage_v=" << voltage_v
                      << std::fixed << std::setprecision(6)
                      << " q_rotor=" << state.q
                      << " dq_rotor=" << state.dq
                      << " tau_rotor=" << state.tau;
        }
        std::cout << '\n';
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::cout << "summary successes=" << successes << "/20\n";
    return successes == 20 ? 0 : 1;
}
